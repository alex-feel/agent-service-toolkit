# app/utils.py
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.aisa.config import RUNNABLE_CONFIGS


async def custom_list_reducer(current: List[str], update) -> List[str]:
    """
    Reduces a list of strings based on the update.
    Custom reducer for List[str].

    This reducer handles updates to a list of strings. It supports:
    - Clearing the list: If the update is a dict with {"action": "clear"}, it returns an empty list.
    - Appending a single string: If the update is a string, it appends it to the list.
    - Extending with a list of strings: If the update is a list of strings, it extends the current list.
    - Raises ValueError for unsupported update types.
    """
    if isinstance(update, dict) and update.get('action') == 'clear':
        return []
    # If update is a single string, append it to the list.
    elif isinstance(update, str):
        return current + [update]
    # If update is a list of strings, extend the current list.
    elif isinstance(update, list) and all(isinstance(x, str) for x in update):
        return current + update
    else:
        raise ValueError('Unsupported update type for custom_list_reducer')


async def merge_consecutive_messages(*messages_lists: Union[List[BaseMessage], Iterable[BaseMessage]]) -> List[BaseMessage]:
    """
    Merges consecutive messages of the same type (AIMessage, HumanMessage, SystemMessage)
    into a single message.

    Args:
        *messages_lists: One or more lists of BaseMessage objects or individual BaseMessage objects.

    Returns:
        A new list of BaseMessage objects with consecutive messages of the same type merged.

    Raises:
        TypeError: If any message in the input is not a BaseMessage or a subclass of it.
        ValueError: If any message content is not a string when attempting to merge.
    """
    # Combine all message lists into one message list
    all_messages: List[BaseMessage] = []  # type: ignore[arg-type]
    for msg_list in messages_lists:
        if isinstance(msg_list, (list, Iterable)):
            all_messages.extend(msg_list)
        else:
            all_messages.append(msg_list)  # type: ignore[arg-type]

    if not all_messages:
        return []

    for msg in all_messages:
        if not isinstance(msg, BaseMessage):
            raise TypeError(f"Expected BaseMessage, got {type(msg)}")

    merged = []
    prev_msg = all_messages[0]

    for current_msg in all_messages[1:]:
        if type(current_msg) is type(prev_msg) and isinstance(current_msg, (AIMessage, HumanMessage, SystemMessage)):
            if not isinstance(prev_msg.content, str):
                raise ValueError(f"Expected string content in previous message, got {
                                 type(prev_msg.content)}")
            if not isinstance(current_msg.content, str):
                raise ValueError(f"Expected string content in current message, got {
                                 type(current_msg.content)}")

            if isinstance(prev_msg.content, str) and isinstance(current_msg.content, str):
                new_content = f"{prev_msg.content}\n\n{current_msg.content}"
                merged_msg = prev_msg.model_copy(
                    update={'content': new_content})
                prev_msg = merged_msg
            else:
                merged.append(prev_msg)
                prev_msg = current_msg
        else:
            merged.append(prev_msg)
            prev_msg = current_msg

    merged.append(prev_msg)
    return merged


async def extract_content_from_messages(messages: list[Union[dict, BaseMessage]], default: str = '') -> str:
    """
    Extracts the content from a list of messages, handling both dictionaries and BaseMessage objects.

    Args:
        messages (list[Union[dict, BaseMessage]]): A list of messages, which can be either dicts or BaseMessage objects.
        default (str, optional): The default string to return if the messages list is empty. Defaults to ''.

    Returns:
        str: The extracted content from the last message in the list.
    """
    if not messages:
        return default

    last_message = messages[-1]

    # Get content from message
    if isinstance(last_message, dict):
        raw_content = last_message.get('content', '')
    else:
        raw_content = last_message.content

    # Handle array case
    if isinstance(raw_content, list):
        return '\n'.join(raw_content)

    return str(raw_content)


async def format_numbered_list_with_header(header: str, items: list[str]) -> str:
    """
    Formats a list of items into a numbered list with a header.

    Args:
        header (str): The header text to prepend to the list
        items (list[str]): List of items to number

    Returns:
        str: Formatted string with header and numbered items
    """
    if not items:
        return header.strip()  # Return just the header if no items

    # Create numbered list items
    numbered_items = [f"{i + 1}. {item}" for i, item in enumerate(items)]

    # Combine header and numbered list
    return f"{header.strip()}\n\n" + '\n'.join(numbered_items)


async def format_section_with_numbered_headers(
        header: str,
        items: list[str],
        item_header_template: str = '# {item_type} {index}\n\n{content}'
) -> str:
    """
    Formats a list of items into a section with numbered headers.

    Args:
        header (str): The header text to prepend to the section.
        items (list[str]): List of items to include in the section.
        item_header_template (str, optional): Template for each item's header.
            Must contain {item_type}, {index}, and {content} placeholders.
            Defaults to "# {item_type} {index}\n\n{content}".

    Returns:
        str: Formatted string with the section header and numbered item headers.
    """
    if not items:
        return header.strip()

    section = [header.strip()]
    for i, item in enumerate(items, 1):
        section.append(
            item_header_template.format(
                item_type='Item',
                index=i,
                content=item.strip()
            )
        )

    return '\n\n'.join(section)


async def get_runnable_config(config_name: str, config: Optional[dict] = None) -> RunnableConfig:
    """
    Retrieves a RunnableConfig based on a given configuration name and an optional override configuration.

    This function fetches a base configuration from the `RUNNABLE_CONFIGS` dictionary using the provided `config_name`.
    If the `config_name` is not found, it defaults to the 'default' configuration.
    It then merges this base configuration with any additional configuration provided in the `config` parameter.
    The merging process ensures that the 'configurable' section of the configurations is also merged.

    Args:
        config_name (str): The name of the configuration to retrieve from `RUNNABLE_CONFIGS`.
        config (Optional[dict]): An optional dictionary containing configuration overrides.

    Returns:
        RunnableConfig: A merged RunnableConfig object.
    """
    # Get base config
    base_config = RUNNABLE_CONFIGS.get(
        config_name, RUNNABLE_CONFIGS['default'])

    # Convert to regular dictionary
    base_dict = dict(base_config)

    # Merge with passed config
    if config:
        merged = {
            **base_dict,
            **config,
            'configurable': {
                **base_dict.get('configurable', {}),
                **config.get('configurable', {})
            }
        }
        return RunnableConfig(**merged)

    return RunnableConfig(**base_dict)
