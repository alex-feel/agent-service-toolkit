# agent.py
from enum import Enum
from typing import Annotated
from typing import Literal
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.types import Command
from langgraph.types import interrupt


load_dotenv()


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class YesNoEnum(str, Enum):
    YES = 'yes'
    NO = 'no'


model = init_chat_model()

checkpointer = MemorySaver()

validate_request_scope_config = RunnableConfig(
    tags=['langsmith:nostream'],
    configurable={
        'model': 'gpt-4o-mini',
        'temperature': 0.5,
        'max_tokens': None,
        'timeout': None,
    }
)

provide_cat_info_config = RunnableConfig(
    configurable={
        'model': 'gpt-4o-mini',
        'temperature': 0.5,
        'max_tokens': None,
        'timeout': None,
    }
)


async def validate_request_scope(state: State) -> Command[Literal['provide_cat_info', 'handle_off_topic']]:
    prompt = ChatPromptTemplate.from_messages([
        ('system', '''
You are the input controller for a "cats-only" system.
Determine if a user request EXCLUSIVELY relates to cats.

Answer ONLY "yes" or "no" (all lower case):
- "yes" if the request is about cats
- "no" otherwise
            '''),
        ('human', '''
Please validate the following user query:
{user_query}
            ''')
    ])

    chain = prompt | model | StrOutputParser()

    result = (chain.invoke(
        {
            'user_query': state['messages'][-1].content,
        },
        config=validate_request_scope_config,
    )).strip().lower()

    try:
        validation_result = YesNoEnum(result)
    except ValueError:
        validation_result = YesNoEnum.YES

    # Decision logic
    if validation_result is YesNoEnum.YES:
        goto = 'provide_cat_info'
    else:
        goto = 'handle_off_topic'

    return Command(
        goto=goto,
    )


async def handle_off_topic(state: State) -> Command[Literal['validate_request_scope']]:  # noqa
    response = interrupt(
        'I can only help with cats. Sorry, but your request is off-topic.')

    return Command(
        update={
            'messages': [HumanMessage(content=response)],
        },
        goto='validate_request_scope',
    )


async def provide_cat_info(state: State):  # noqa
    prompt = ChatPromptTemplate.from_messages([
        ('system', '''
    You are an expert on cats.
    You only answer questions that are strictly about cats.

    Always reply concisely and helpfully.
            '''),
        ('human', '''
    {user_query}
            ''')
    ])

    chain = prompt | model

    result = chain.invoke(
        {
            'user_query': state['messages'][-1].content,
        },
        config=provide_cat_info_config,
    )

    return {
        'messages': [AIMessage(content=result)],
    }


async def get_user_feedback(state: State) -> Command[Literal['provide_cat_info']]:  # noqa
    response = interrupt('The cat info is ready. If you have any comments or additional details, '
                         'please send them to me and I will continue working.')

    return Command(
        update={
            'messages': [HumanMessage(content=response)]
        },
        goto='provide_cat_info',
    )


workflow = StateGraph(State)

workflow.add_node('validate_request_scope', validate_request_scope)
workflow.add_node('handle_off_topic', handle_off_topic)
workflow.add_node('provide_cat_info', provide_cat_info)
workflow.add_node('get_user_feedback', get_user_feedback)

workflow.add_edge(START, 'validate_request_scope')
workflow.add_edge('provide_cat_info', 'get_user_feedback')

cats_graph = workflow.compile(
    checkpointer=checkpointer,
)
