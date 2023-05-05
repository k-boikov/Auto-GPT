import copy
import json
from typing import Dict, List, Tuple

from autogpt.agent import Agent
from autogpt.config import Config
from autogpt.llm import count_message_tokens, count_string_tokens
from autogpt.llm.llm_utils import create_chat_completion
from autogpt.log_cycle.log_cycle import PROMPT_SUMMARY_FILE_NAME, SUMMARY_FILE_NAME

cfg = Config()


def get_newly_trimmed_messages(
    full_message_history: List[Dict[str, str]],
    current_context: List[Dict[str, str]],
    last_memory_index: int,
) -> Tuple[List[Dict[str, str]], int]:
    """
    This function returns a list of dictionaries contained in full_message_history
    with an index higher than prev_index that are absent from current_context.

    Args:
        full_message_history (list): A list of dictionaries representing the full message history.
        current_context (list): A list of dictionaries representing the current context.
        last_memory_index (int): An integer representing the previous index.

    Returns:
        list: A list of dictionaries that are in full_message_history with an index higher than last_memory_index and absent from current_context.
        int: The new index value for use in the next loop.
    """
    # Select messages in full_message_history with an index higher than last_memory_index
    new_messages = [
        msg for i, msg in enumerate(full_message_history) if i > last_memory_index
    ]

    # Remove messages that are already present in current_context
    new_messages_not_in_context = [
        msg for msg in new_messages if msg not in current_context
    ]

    # Find the index of the last message processed
    new_index = last_memory_index
    if new_messages_not_in_context:
        last_message = new_messages_not_in_context[-1]
        new_index = full_message_history.index(last_message)

    return new_messages_not_in_context, new_index


def update_running_summary(
    agent: Agent,
    current_memory: str,
    new_events: List[Dict[str, str]],
    token_limit: int,
) -> Dict[str, str]:
    """
    This function takes a list of dictionaries representing new events and combines them with the current summary,
    focusing on key and potentially important information to remember. The updated summary is returned in a message
    formatted in the 1st person past tense.

    Args:
        agent:
        current_memory:
        token_limit:
        new_events (List[Dict]): A list of dictionaries containing the latest events to be added to the summary.

    Returns:
        Dict[str, str]: A message containing the updated summary of actions, formatted in the 1st person past tense.

    Example:
        new_events = [{"event": "entered the kitchen."}, {"event": "found a scrawled note with the number 7"}]
        update_running_summary(new_events)
        # Returns: "This reminds you of these events from your past: \nI entered the kitchen and found a scrawled note saying 7."
    """
    # Create a copy of the new_events list to prevent modifying the original list
    new_events = copy.deepcopy(new_events)

    send_token_limit = token_limit - 1000
    current_tokens_used = count_string_tokens(current_memory, cfg.fast_llm_model)
    if current_tokens_used > token_limit:
        # TODO: maybe needs optimization in order to remove older memories first
        current_memory = summarize_text(current_memory)
        current_tokens_used = count_string_tokens(current_memory, cfg.fast_llm_model)

    # Replace "assistant" with "you". This produces much better first person past tense results.
    for i, event in enumerate(new_events):
        tokens_to_add = count_message_tokens([event], cfg.fast_llm_model)
        if event["role"].lower() == "assistant":
            event["role"] = "you"

            # Remove "thoughts" dictionary from "content"
            try:
                content_dict = json.loads(event["content"])
            except:
                continue
            if "thoughts" in content_dict:
                del content_dict["thoughts"]
            event["content"] = json.dumps(content_dict)

        elif event["role"].lower() == "system":
            event["role"] = "your computer"

        # Delete all user messages
        elif event["role"] == "user":
            new_events.remove(event)
            continue

        # Commands` result can pass the immediate test for length but in combination
        # with other events they might exceed the token limit.
        if current_tokens_used + tokens_to_add > send_token_limit:
            # Check if the main model is not a lot bigger, rare case, but might happen
            # We sill just skip such massive events for now
            if tokens_to_add > send_token_limit:
                # TODO: chunk the message by paragraphs and then summarize or use some NLTK
                new_events.remove(event)
                continue
            # Merge the old memory with the new events looped so far
            current_memory = summarize_events(
                agent, current_memory, new_events[0 : i - 1], token_limit
            )
            # Process the rest of the new events with the newly summarized current memory
            return update_running_summary(
                agent, current_memory, new_events[i:], token_limit
            )

        current_tokens_used += tokens_to_add

    # This can happen at any point during execution, not just the beginning
    if len(new_events) == 0:
        new_events = "Nothing new happened."

    current_memory = summarize_events(agent, current_memory, new_events, token_limit)

    message_to_return = {
        "role": "system",
        "content": f"This reminds you of these events from your past: \n{current_memory}",
    }

    return message_to_return


def summarize_text(text: str):
    """
    This function takes text and summarizes it using LLM.

    Args:
        text: (str): The text to summarize

    Returns:
        str: The summarized text
    """

    prompt = f'''Summarize the following text in a neutral and objective manner, without adding any personal opinions or interpretations. If not possible, just return keywords or cut text you think is irrelevant.

Text to summarize:
"""
{text}
"""
'''

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return create_chat_completion(messages, cfg.fast_llm_model)


def summarize_events(
    agent: Agent,
    current_memory: str,
    new_events: List[Dict[str, str]],
    token_limit: int,
):
    prompt = f'''Your task is to create a concise running summary of actions and information results in the provided text, focusing on key and potentially important information to remember.

You will receive the current summary and the your latest actions. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.

Summary So Far:
"""
{current_memory}
"""

Latest Development:
"""
{new_events}
"""
'''

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    summary_tlength = count_message_tokens(messages, cfg.fast_llm_model)
    if summary_tlength > token_limit:
        # This should never happen in theory since we check both old memory and new events
        return current_memory

    agent.log_cycle_handler.log_cycle(
        agent.config.ai_name,
        agent.created_at,
        agent.cycle_count,
        messages,
        PROMPT_SUMMARY_FILE_NAME,
    )

    current_memory = create_chat_completion(messages, cfg.fast_llm_model)

    agent.log_cycle_handler.log_cycle(
        agent.config.ai_name,
        agent.created_at,
        agent.cycle_count,
        current_memory,
        SUMMARY_FILE_NAME,
    )

    return current_memory
