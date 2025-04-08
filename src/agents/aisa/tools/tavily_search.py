# app/tools/tavily_search.py
from typing import Literal

from langchain_core.tools.structured import StructuredTool
from tavily import AsyncTavilyClient
from tavily import TavilyClient


async def tavily_search_async(
        query: str,
        search_depth: Literal['basic', 'advanced'] = 'basic',
        max_results: int = 5
) -> dict:
    """
    Performs an asynchronous web search using Tavily API.

    Args:
        query: Search query text
        search_depth: Search depth mode ('basic'|'advanced')
        max_results: Maximum number of results to return

    Returns:
        dict: Search results with metadata

    Raises:
        Exception: On API errors
    """
    try:
        client = AsyncTavilyClient()  # Async client работает с контекстным менеджером
        return await client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_raw_content=True,
            topic='general'
        )
    except Exception as e:
        raise Exception(f"Tavily search failed: {str(e)}")


def tavily_search_sync(
        query: str,
        search_depth: Literal['basic', 'advanced'] = 'basic',
        max_results: int = 5
) -> dict:
    """
    Performs a synchronous web search using Tavily API.

    Args:
        query: Search query text
        search_depth: Search depth mode ('basic'|'advanced')
        max_results: Maximum number of results to return

    Returns:
        dict: Search results with metadata

    Raises:
        Exception: On API errors
    """
    try:
        client = TavilyClient()  # Синхронный клиент без контекстного менеджера
        return client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_raw_content=True,
            topic='general'
        )
    except Exception as e:
        raise Exception(f"Tavily search failed: {str(e)}")


def get_tavily_tool(mode: Literal['sync', 'async'] = 'async') -> StructuredTool:
    """Factory for Tavily search tools"""
    description = '''
    Powerful web search tool using Tavily API.
    Input: single search query string.
    Parameters:
    - search_depth: basic/advanced (default basic)
    - max_results: 1-10 (default 5)
    Returns rich results with titles, URLs, content snippets and metadata.
    '''

    if mode == 'async':
        return StructuredTool.from_function(
            coroutine=tavily_search_async,
            name='tavily_search_async',
            description=description.strip()
        )

    return StructuredTool.from_function(
        func=tavily_search_sync,
        name='tavily_search_sync',
        description=description.strip()
    )


# Optionally expose direct tool instances
tavily_tool_async = get_tavily_tool('async')
tavily_tool_sync = get_tavily_tool('sync')
