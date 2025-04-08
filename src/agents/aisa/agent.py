# app/agent.py
import asyncio
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import RemoveMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langgraph.types import interrupt
from langmem import create_search_memory_tool

from agents.aisa.config import GAPS_SEARCH_RECURSION_LIMIT
from agents.aisa.config import GET_FEEDBACK_ON_REQUIREMENTS_INTERRUPT_MESSAGE
from agents.aisa.config import MAX_LLM_OUTPUT_LIST_ELEMENTS
from agents.aisa.config import OFF_TOPIC_INTERRUPT_MESSAGE
from agents.aisa.parsers import LineListOutputParser
from agents.aisa.prompts import GAPS_ANALYSIS_SYSTEM_PROMPT
from agents.aisa.prompts import REQUIREMENTS_REWRITING_SYSTEM_PROMPT
from agents.aisa.prompts import REQUIREMENTS_WRITING_SYSTEM_PROMPT
from agents.aisa.prompts import TERM_DEFINITION_SEARCH_HUMAN_PROMPT
from agents.aisa.prompts import TERM_DEFINITION_SEARCH_SYSTEM_PROMPT
from agents.aisa.prompts import TERM_EXTRACTION_SYSTEM_PROMPT
from agents.aisa.prompts import TOPIC_RESEARCH_HUMAN_PROMPT
from agents.aisa.prompts import TOPIC_RESEARCH_SYSTEM_PROMPT
from agents.aisa.prompts import USER_FEEDBACK_ANALYSIS_SYSTEM_PROMPT
from agents.aisa.prompts import USER_FEEDBACK_INTENT_ANALYSIS_SYSTEM_PROMPT
from agents.aisa.prompts import USER_QUERY_ANALYSIS_SYSTEM_PROMPT
from agents.aisa.prompts import USER_QUERY_VALIDATION_HUMAN_PROMPT
from agents.aisa.prompts import USER_QUERY_VALIDATION_SYSTEM_PROMPT
from agents.aisa.state import FeedbackIntentEnum
from agents.aisa.state import OverallState
from agents.aisa.state import PublicState
from agents.aisa.state import SingleTermDefinitionState
from agents.aisa.state import SingleTopicState
from agents.aisa.state import YesNoEnum
from agents.aisa.tools.spotware_help_centre_search import spotware_help_centre_tool_sync
from agents.aisa.tools.tavily_search import tavily_tool_sync
from agents.aisa.utils import extract_content_from_messages
from agents.aisa.utils import format_numbered_list_with_header
from agents.aisa.utils import format_section_with_numbered_headers
from agents.aisa.utils import get_runnable_config
from agents.aisa.utils import merge_consecutive_messages

load_dotenv()

long_memory_store = InMemoryStore(
    index={
        'dims': 1536,
        'embed': 'openai:text-embedding-3-small',
    }
)

model = init_chat_model()

checkpointer = MemorySaver()


async def validate_request_scope(state: OverallState, config=None) -> Command[Literal[
    'handle_off_topic', 'extract_terms_from_user_query']
]:
    config = await get_runnable_config('classification_config', config)

    prompt = ChatPromptTemplate.from_messages([
        USER_QUERY_VALIDATION_SYSTEM_PROMPT,
        USER_QUERY_VALIDATION_HUMAN_PROMPT,
    ])

    chain = prompt | model | StrOutputParser()

    result = await chain.ainvoke(
        {
            'user_query': state['messages'][-1].content,
        },
        config=config,
    )

    res = result.strip().lower()

    # Validate and parse response
    try:
        validation_result = YesNoEnum(res)
    except ValueError:
        validation_result = YesNoEnum.YES

    # Decision logic
    if validation_result is YesNoEnum.YES:
        goto = 'extract_terms_from_user_query'
        messages = [AIMessage(content='Extracting terms...')]

        # In case the previous run was cancelled we need to reset the context (internal_messages)
        # Get all current internal messages from the state
        internal_messages = state.get('internal_messages', [])

        if len(internal_messages) > 0:
            # Generate a list of RemoveMessage objects for all messages
            internal_messages_with_removals = [
                RemoveMessage(id=msg.id) for msg in internal_messages]
            # Append a new message to the list
            internal_messages_with_removals.append(
                state['messages'][-1])
            internal_messages = internal_messages_with_removals
        else:
            internal_messages = [state['messages'][-1]]

    elif validation_result is YesNoEnum.NO:
        goto = 'handle_off_topic'
        messages = []
        internal_messages = []

    else:
        goto = None
        messages = None
        internal_messages = None

    return Command(
        update={
            'messages': messages,
            'internal_messages': internal_messages,
        },
        goto=goto,
    )


async def handle_off_topic(state: OverallState) -> Command[Literal['validate_request_scope']]:  # noqa
    new_query = interrupt(OFF_TOPIC_INTERRUPT_MESSAGE)

    return Command(
        update={
            'messages': [HumanMessage(content=new_query)],
        },
        goto='validate_request_scope',
    )


async def extract_terms_from_user_query(state: OverallState, config=None):
    config = await get_runnable_config('extraction_config', config)

    prompt = ChatPromptTemplate.from_messages([
        TERM_EXTRACTION_SYSTEM_PROMPT,
        state['internal_messages'][0],
    ])

    chain = prompt | model | LineListOutputParser()

    result = await chain.ainvoke(
        {},
        config=config
    )

    messages = [AIMessage(content='Collecting term definitions...')]

    return {
        'terms': result,
        'messages': messages,
    }


# Term Definition Search
async def search_term_definition(state: SingleTermDefinitionState, config=None):
    config = await get_runnable_config('agent_config', config)

    agent = create_react_agent(
        model=model,
        tools=[
            create_search_memory_tool(
                namespace=('aisa-memories',),
                store=long_memory_store,
            ),
            spotware_help_centre_tool_sync,
            tavily_tool_sync,
        ],
        version='v2',
    )

    prompt = ChatPromptTemplate.from_messages([
        TERM_DEFINITION_SEARCH_SYSTEM_PROMPT,
        TERM_DEFINITION_SEARCH_HUMAN_PROMPT,
    ])

    chain = prompt | agent

    result = await chain.ainvoke(
        {
            'user_query': state['user_query'],
            'search_term': state['search_term'],
        },
        config=config,
    )

    messages = [msg for msg in result['messages']]
    definition = await extract_content_from_messages(messages, 'No definition')

    return {
        'definition': definition,
    }


search_term_definition_workflow = StateGraph(SingleTermDefinitionState)

search_term_definition_workflow.add_node(
    'Term Definition Search', search_term_definition)

search_term_definition_workflow.add_edge(START, 'Term Definition Search')
search_term_definition_workflow.add_edge('Term Definition Search', END)

search_term_definition_graph = search_term_definition_workflow.compile(
    name='Term Definition Search Engine',
)


async def collect_term_definitions(state: OverallState, config=None):
    config = config or {}

    # Forming a list of coroutines
    coroutines = []
    for term in state['terms']:
        coroutines.append(
            search_term_definition_graph.ainvoke(
                {
                    'user_query': state['internal_messages'][0].content,
                    'search_term': term,
                },
                config=config,
            )
        )

    # Run it all at once and wait for all the results
    results = await asyncio.gather(*coroutines)

    # Collect the contexts
    context = [res['definition'] for res in results]

    result_with_header = await format_section_with_numbered_headers(
        'Here are definitions, related to the user query:',
        context,
        item_header_template='# Term/concept/entity {index}\n\n{content}'
    )

    internal_messages = [AIMessage(content=result_with_header)]

    messages = [
        AIMessage(
            content='Term definitions collection finished. Analyzing your query...'),
    ]

    return {
        'messages': messages,
        'internal_messages': internal_messages,
    }


async def analyze_user_query(state: OverallState, config=None):
    config = await get_runnable_config('thought_config', config)

    merged_messages = await merge_consecutive_messages(
        state['internal_messages'],
        [HumanMessage(
            content='Please analyze the user initial query, the dialog history, and provide the questions/topics.'
        )],
    )

    prompt = ChatPromptTemplate.from_messages([
        USER_QUERY_ANALYSIS_SYSTEM_PROMPT,
        *merged_messages,
    ])

    chain = prompt | model | LineListOutputParser()

    result = await chain.ainvoke(
        {
            'max_results': MAX_LLM_OUTPUT_LIST_ELEMENTS,
        },
        config=config
    )

    result_with_header = await format_numbered_list_with_header(
        'Based on the initial query, the following questions/topics are identified to be researched:',
        result
    )

    internal_messages = [AIMessage(content=result_with_header)]

    messages = [
        AIMessage(
            content='Query analysis finished. Researching the questions/topics and collecting context...'),
    ]

    return {
        'messages': messages,
        'internal_messages': internal_messages,
        'research_topics': result,
        'context_collection_initiator': 'USER_QUERY_ANALYSIS',
    }


# Topic Research
async def research_topic(state: SingleTopicState, config=None):
    config = await get_runnable_config('agent_config', config)

    agent = create_react_agent(
        model=model,
        tools=[
            create_search_memory_tool(
                namespace=('aisa-memories',),
                store=long_memory_store,
            ),
            spotware_help_centre_tool_sync,
            tavily_tool_sync,
        ],
        version='v2',
    )

    prompt = ChatPromptTemplate.from_messages([
        TOPIC_RESEARCH_SYSTEM_PROMPT,
        TOPIC_RESEARCH_HUMAN_PROMPT,
    ])

    chain = prompt | agent

    result = await chain.ainvoke(
        {
            'user_query': state['user_query'],
            'research_topic': state['research_topic'],
        },
        config=config,
    )

    messages = [msg for msg in result['messages']]
    context = await extract_content_from_messages(messages, 'No context')

    return {
        'context': context,
    }


research_topic_workflow = StateGraph(SingleTopicState)

research_topic_workflow.add_node(
    'Topic Research', research_topic)

research_topic_workflow.add_edge(START, 'Topic Research')
research_topic_workflow.add_edge('Topic Research', END)

research_topic_graph = research_topic_workflow.compile(
    name='Topic Research Engine',
)


async def collect_context(state: OverallState, config=None) -> Command[Literal[
    'analyze_context_for_gaps', 'write_requirements', 'rewrite_requirements']
]:
    config = config or {}

    # Forming a list of coroutines
    coroutines = []
    for topic in state['research_topics']:
        coroutines.append(
            research_topic_graph.ainvoke(
                {
                    'user_query': state['internal_messages'][0].content,
                    'research_topic': topic,
                },
                config=config,
            )
        )

    # Run it all at once and wait for all the results
    results = await asyncio.gather(*coroutines)

    # Collect the contexts
    context = [res['context'] for res in results]

    result_with_header = await format_section_with_numbered_headers(
        'Based on the research questions/topics, the following context was found:',
        context,
        item_header_template='# Context {index}\n\n{content}'
    )

    current_count = state.get('gaps_search_count', 0)
    initiator = state['context_collection_initiator']
    messages = []

    if current_count < GAPS_SEARCH_RECURSION_LIMIT:
        goto_node = 'analyze_context_for_gaps'
        messages.append(AIMessage(
            content='Context collection finished. Analyzing context for gaps...'))
    elif initiator == 'USER_QUERY_ANALYSIS':
        goto_node = 'write_requirements'
        messages.append(
            AIMessage(content='Context collection finished. Writing requirements...'))
    elif initiator == 'USER_FEEDBACK_ANALYSIS':
        goto_node = 'rewrite_requirements'
        messages.append(
            AIMessage(content='Context collection finished. Rewriting requirements...'))
    else:
        goto_node = None
        messages = None

    return Command(
        update={
            'messages': messages,
            'internal_messages': [AIMessage(content=result_with_header)],
        },
        goto=goto_node,
    )


async def analyze_context_for_gaps(state: OverallState, config=None):
    config = await get_runnable_config('thought_config', config)

    merged_messages = await merge_consecutive_messages(
        state['internal_messages'],
        [HumanMessage(
            content='Please analyze the user initial query, the dialog history, and provide the questions/topics.'
        )],
    )

    prompt = ChatPromptTemplate.from_messages([
        GAPS_ANALYSIS_SYSTEM_PROMPT,
        *merged_messages,
    ])

    chain = prompt | model | LineListOutputParser()

    result = await chain.ainvoke(
        {
            'max_results': MAX_LLM_OUTPUT_LIST_ELEMENTS,
        },
        config=config
    )

    result_with_header = await format_numbered_list_with_header(
        'Having analyzed the initial query and dialog history, '
        'the following questions/topics are identified to be further researched:',
        result
    )

    internal_messages = [AIMessage(content=result_with_header)]

    messages = [
        AIMessage(
            content='Context gaps analysis finished. Researching the questions/topics and collecting context...'),
    ]

    return {
        'messages': messages,
        'internal_messages': internal_messages,
        'research_topics': result,
        'gaps_search_count': 1,
    }


async def write_requirements(state: OverallState, config=None):
    config = await get_runnable_config('requirements_writing_config', config)

    merged_messages = await merge_consecutive_messages(
        state['internal_messages'],
    )

    prompt = ChatPromptTemplate.from_messages([
        REQUIREMENTS_WRITING_SYSTEM_PROMPT,
        *merged_messages,
    ])

    chain = prompt | model

    result = await chain.ainvoke(
        {},
        config=config
    )

    # Add title
    header = 'Here are the initial draft requirements based on all the context collected:\n'

    # Number list items
    result_with_header = f"{header}\n{result.content}"

    return {
        'messages': result,
        'internal_messages': [AIMessage(content=result_with_header)],
    }


async def get_user_feedback(state: OverallState) -> Command[Literal['analyze_user_feedback_intent']]:  # noqa
    feedback = interrupt(GET_FEEDBACK_ON_REQUIREMENTS_INTERRUPT_MESSAGE)

    messages = [HumanMessage(content=feedback)]

    return Command(
        update={
            'messages': messages,
            'internal_messages': messages,
        },
        goto='analyze_user_feedback_intent',
    )


async def analyze_user_feedback_intent(state: OverallState, config=None) -> Command[Literal[
    'analyze_user_feedback', 'rewrite_requirements']
]:
    config = await get_runnable_config('classification_config', config)

    merged_messages = await merge_consecutive_messages(
        state['internal_messages'],
    )

    prompt = ChatPromptTemplate.from_messages([
        USER_FEEDBACK_INTENT_ANALYSIS_SYSTEM_PROMPT,
        *merged_messages,
    ])

    chain = prompt | model | StrOutputParser()

    result = await chain.ainvoke(
        {},
        config=config,
    )

    res = result.strip().lower()

    # Validate and parse response
    try:
        intent_analysis_result = FeedbackIntentEnum(res)
    except ValueError:
        intent_analysis_result = FeedbackIntentEnum.APPROVE

    # Decision logic
    if intent_analysis_result is FeedbackIntentEnum.COLLECT_CONTEXT:
        goto = 'analyze_user_feedback'
        messages = AIMessage(content='Analyzing the feedback...')

    elif intent_analysis_result is FeedbackIntentEnum.REWRITE:
        goto = 'rewrite_requirements'
        messages = AIMessage(content='Rewriting requirements...')

    elif intent_analysis_result is FeedbackIntentEnum.APPROVE:
        goto = 'get_user_feedback'
        messages = []

    else:
        goto = None
        messages = None

    return Command(
        update={
            'messages': messages,
        },
        goto=goto,
    )


async def analyze_user_feedback(state: OverallState, config=None):
    config = await get_runnable_config('thought_config', config)

    merged_messages = await merge_consecutive_messages(
        state['internal_messages'],
    )

    prompt = ChatPromptTemplate.from_messages([
        USER_FEEDBACK_ANALYSIS_SYSTEM_PROMPT,
        *merged_messages,
    ])

    chain = prompt | model | LineListOutputParser()

    result = await chain.ainvoke(
        {
            'max_results': MAX_LLM_OUTPUT_LIST_ELEMENTS,
        },
        config=config
    )

    result_with_header = await format_numbered_list_with_header(
        'Based on the feedback, the following questions/topics are identified to be researched:',
        result
    )

    internal_messages = [AIMessage(content=result_with_header)]

    messages = [
        AIMessage(
            content='Feedback analysis finished. Researching the questions/topics and collecting context...'),
    ]

    gaps_search_count = 0 if GAPS_SEARCH_RECURSION_LIMIT == 0 else -1

    return {
        'messages': messages,
        'internal_messages': internal_messages,
        'research_topics': result,
        'gaps_search_count': gaps_search_count,
        'context_collection_initiator': 'USER_FEEDBACK_ANALYSIS',
    }


async def rewrite_requirements(state: OverallState, config=None):
    config = await get_runnable_config('requirements_writing_config', config)

    merged_messages = await merge_consecutive_messages(
        state['internal_messages'],
    )

    prompt = ChatPromptTemplate.from_messages([
        REQUIREMENTS_REWRITING_SYSTEM_PROMPT,
        *merged_messages,
    ])

    chain = prompt | model

    result = await chain.ainvoke(
        {},
        config=config
    )

    # Add title
    header = 'Here are the updated draft requirements based on the feedback and all the context collected:\n'

    # Number list items
    result_with_header = f"{header}\n{result.content}"

    return {
        'messages': result,
        'internal_messages': [AIMessage(content=result_with_header)],
    }


workflow = StateGraph(OverallState, input=PublicState, output=PublicState)

workflow.add_node('validate_request_scope', validate_request_scope)
workflow.add_node('handle_off_topic', handle_off_topic)
workflow.add_node('extract_terms_from_user_query',
                  extract_terms_from_user_query)
workflow.add_node('collect_term_definitions', collect_term_definitions)
workflow.add_node('analyze_user_query', analyze_user_query)
workflow.add_node('collect_context', collect_context)
workflow.add_node('analyze_context_for_gaps', analyze_context_for_gaps)
workflow.add_node('write_requirements', write_requirements)
workflow.add_node('get_user_feedback', get_user_feedback)
workflow.add_node('analyze_user_feedback_intent', analyze_user_feedback_intent)
workflow.add_node('analyze_user_feedback', analyze_user_feedback)
workflow.add_node('rewrite_requirements', rewrite_requirements)

workflow.add_edge(START, 'validate_request_scope')
workflow.add_edge('extract_terms_from_user_query', 'collect_term_definitions')
workflow.add_edge('collect_term_definitions', 'analyze_user_query')
workflow.add_edge('analyze_user_query', 'collect_context')
workflow.add_edge('analyze_context_for_gaps', 'collect_context')
workflow.add_edge('write_requirements', 'get_user_feedback')
workflow.add_edge('analyze_user_feedback', 'collect_context')
workflow.add_edge('rewrite_requirements', 'get_user_feedback')

aisa_graph = workflow.compile(
    checkpointer=checkpointer,
)
