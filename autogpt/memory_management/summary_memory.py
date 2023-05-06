import copy
import json
from typing import Dict, List, Tuple

from autogpt.agent import Agent
from autogpt.config import Config
from autogpt.llm import Message, count_message_tokens
from autogpt.llm.llm_utils import create_chat_completion
from autogpt.log_cycle.log_cycle import PROMPT_SUMMARY_FILE_NAME, SUMMARY_FILE_NAME
from autogpt.processing.text import split_text

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

    if len(new_messages_not_in_context) > 2:
        # Trap strange behavior
        print("new_messages_not_in_context")

    # Find the index of the last message processed
    new_index = last_memory_index
    if new_messages_not_in_context:
        last_message = new_messages_not_in_context[-1]
        new_index = full_message_history.index(last_message)

    return new_messages_not_in_context, new_index


def update_running_summary(
    agent: Agent,
    current_memory: Message,
    new_events: List[Message],
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

    # This can happen at any point during execution, not just the beginning
    if len(new_events) == 0:
        return current_memory

    # Create a copy of the new_events list to prevent modifying the original list
    new_events = copy.deepcopy(new_events)

    send_token_limit = token_limit - count_message_tokens(
        get_message_for_summarization(), cfg.fast_llm_model
    )
    current_tokens_used = count_message_tokens([current_memory], cfg.fast_llm_model)

    tokens_to_add = 0
    i = 0
    it = iter(new_events)
    event = next(it, None)
    while event:
        # Replace "assistant" with "you". This produces much better first person past tense results.
        if event["role"].lower() == "assistant":
            event["role"] = "you"

            # Remove "thoughts" dictionary from "content"
            try:
                content_dict = json.loads(event["content"])
            except:
                event = next(it, None)
                i += 1
                continue
            if "thoughts" in content_dict:
                del content_dict["thoughts"]
            event["content"] = json.dumps(content_dict)
            tokens_to_add = count_message_tokens([event], cfg.fast_llm_model)

        elif event["role"].lower() == "system":
            event["role"] = "your computer"
            tokens_to_add = count_message_tokens([event], cfg.fast_llm_model)

        # Delete all user messages
        elif event["role"] == "user":
            new_events.remove(event)
            event = next(it, None)
            i += 1
            continue

        current_tokens_used += tokens_to_add

        # Commands` result can pass the immediate test for length but in combination
        # with other events they might exceed the token limit.
        if current_tokens_used > send_token_limit:
            # Check if the main model is not a lot bigger, rare case, but might happen
            # In that case a single event would not fit for summarization
            if tokens_to_add > send_token_limit:
                tokens_that_would_have_fit = token_limit - (
                    current_tokens_used - tokens_to_add
                )
                split_content = list(
                    split_text(event["content"], tokens_that_would_have_fit)
                )

                if len(split_content) == 0:
                    # We failed to split, just remove that event
                    new_events.remove(event)
                    current_tokens_used -= tokens_to_add
                    event = next(it, None)
                    i += 1
                    continue
                else:
                    event["content"] = split_content[0]
                    # Merge the old memory with the first part of the chunk
                    new_memory = summarize_events(
                        agent, current_memory, new_events[:i], token_limit
                    )
                    current_memory = current_memory_message(new_memory)

                    if len(split_content) > 1:
                        # Create events from the rest of the chunks
                        events_from_split = [
                            {"role": event["role"], "content": chunk}
                            for chunk in split_content[1:]
                        ]
                    else:
                        events_from_split = []

                    new_events = events_from_split + new_events[i:]
                    it = iter(new_events)
                    event = next(it, None)
                    i = 0
                    current_tokens_used = count_message_tokens(
                        [current_memory], cfg.fast_llm_model
                    )
                    continue

            # Merge the old memory with the new events looped so far
            new_memory = summarize_events(
                agent, current_memory, new_events[0 : i - 1], token_limit
            )
            current_memory = current_memory_message(new_memory)
            # Process the rest of the new events with the newly summarized current memory
            new_events = new_events[i:]
            it = iter(new_events)
            event = next(it, None)
            i = 0
            current_tokens_used = count_message_tokens(
                [current_memory], cfg.fast_llm_model
            )
            continue

        event = next(it, None)
        i += 1

    new_memory = summarize_events(agent, current_memory, new_events, token_limit)

    return current_memory_message(new_memory)


def current_memory_message(memory: str):
    return {
        "role": "system",
        "content": f"This reminds you of these events from your past: \n{memory}",
    }


def get_message_for_summarization(current_memory_content="", new_events=""):
    prompt = f'''Your task is to create a concise running summary of actions and information results in the provided text, focusing on key and potentially important information to remember.

You will receive the current summary and the your latest actions. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.

Summary So Far:
"""
{current_memory_content}
"""

Latest Development:
"""
{new_events}
"""
'''
    return [
        {
            "role": "user",
            "content": prompt,
        }
    ]


def summarize_events(
    agent: Agent,
    current_memory: Message,
    new_events: List[Dict[str, str]],
    token_limit: int,
):
    message = get_message_for_summarization(current_memory["content"], str(new_events))

    summary_tlength = count_message_tokens(message, cfg.fast_llm_model)
    if summary_tlength > token_limit:
        # This should never happen in theory since we summed both old memory and new events
        return current_memory

    agent.log_cycle_handler.log_cycle(
        agent.config.ai_name,
        agent.created_at,
        agent.cycle_count,
        message,
        PROMPT_SUMMARY_FILE_NAME,
    )

    current_memory = create_chat_completion(message, cfg.fast_llm_model)

    agent.log_cycle_handler.log_cycle(
        agent.config.ai_name,
        agent.created_at,
        agent.cycle_count,
        current_memory,
        SUMMARY_FILE_NAME,
    )

    return current_memory
