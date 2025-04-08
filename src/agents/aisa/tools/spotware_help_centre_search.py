# app/tools/spotware_help_centre_search.py
from typing import Literal

import httpx
from langchain_core.tools.structured import StructuredTool


async def spotware_help_centre_search_async(query: str) -> dict:
    """
    Performs an asynchronous search in Spotware Help Centre documentation.

    Args:
        query (str): Search query to find in Help Centre documentation

    Returns:
        dict: API response with search results containing relevant documentation entries

    Raises:
        Exception: If API request fails or returns non-200 status
    """
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                'http://localhost:8080/api/v1/retrievers/help-centre',
                json={'query': query},
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise Exception(f"API request failed with status {
                            e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"API connection error: {str(e)}")


def spotware_help_centre_search_sync(query: str) -> dict:
    """
    Performs a synchronous search in Spotware Help Centre documentation.

    Args:
        query (str): Search query to find in Help Centre documentation

    Returns:
        dict: API response with search results containing relevant documentation entries

    Raises:
        Exception: If API request fails or returns non-200 status
    """
    with httpx.Client(timeout=60) as client:
        try:
            response = client.post(
                'http://localhost:8000/api/v1/retrievers/help-centre',
                json={'query': query},
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise Exception(f"API request failed with status {
                            e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"API connection error: {str(e)}")


def get_spotware_help_centre_tool(mode: Literal['sync', 'async'] = 'async') -> StructuredTool:
    """
    Factory function that returns either synchronous or asynchronous version of the tool

    Args:
        mode: 'sync' or 'async' to select tool version

    Returns:
        StructuredTool: Requested version of the search tool
    """
    common_description = '''
    Use this tool to search Spotware's Help Centre documentation.
    Input should be a search query string.
    Returns relevant documentation entries with content and metadata.
    '''

    if mode == 'async':
        return StructuredTool.from_function(
            coroutine=spotware_help_centre_search_async,
            name='spotware_help_centre_search_async',
            description=common_description.strip()
        )

    return StructuredTool.from_function(
        func=spotware_help_centre_search_sync,
        name='spotware_help_centre_search_sync',
        description=common_description.strip()
    )


# Optionally expose direct tool instances
spotware_help_centre_tool_async = get_spotware_help_centre_tool('async')
spotware_help_centre_tool_sync = get_spotware_help_centre_tool('sync')
