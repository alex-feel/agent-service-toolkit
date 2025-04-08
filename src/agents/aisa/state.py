# app/state.py
from enum import Enum
from operator import add
from typing import Annotated
from typing import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class PublicState(TypedDict):
    """Represents the public state of the application."""
    messages: Annotated[list[AnyMessage], add_messages]


class InternalState(TypedDict):
    """Represents the internal state of the application."""
    terms: list[str]
    context_collection_initiator: str
    research_topics: list[str]
    gaps_search_count: Annotated[int, add]
    internal_messages: Annotated[list[AnyMessage], add_messages]


class OverallState(PublicState, InternalState):
    """Represents the overall state of the application."""
    pass


class SingleTermDefinitionState(TypedDict):
    """Represents the state for a single term definition."""
    user_query: str
    search_term: str
    definition: str


class SingleTopicState(TypedDict):
    """Represents the state for a single research topic."""
    user_query: str
    research_topic: str
    context: str


class YesNoEnum(str, Enum):
    """Represents a yes/no choice."""
    YES = 'yes'
    NO = 'no'


class FeedbackIntentEnum(str, Enum):
    """Represents the intent of feedback."""
    APPROVE = 'approve'
    REWRITE = 'rewrite'
    COLLECT_CONTEXT = 'collect_context'
