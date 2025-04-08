# app/config.py
from langchain_core.runnables.config import RunnableConfig


MAX_LLM_OUTPUT_LIST_ELEMENTS = 1
GAPS_SEARCH_RECURSION_LIMIT = 0

GET_FEEDBACK_ON_REQUIREMENTS_INTERRUPT_MESSAGE = ('The requirements are ready. If you have any comments, remarks '
                                                  'or additional requirements, send them to me and I will '
                                                  'continue working.')

OFF_TOPIC_INTERRUPT_MESSAGE = ('Your request is beyond my capabilities. I specialize exclusively in developing '
                               'business/system requirements for cTrader, analyzing Spotware documentation, '
                               'creating technical specifications for the cTrader ecosystem. Please ask a '
                               'different question within these competencies.')

# https://github.com/langchain-ai/agent-chat-ui?tab=readme-ov-file#hiding-messages-in-the-chat
RUNNABLE_CONFIGS = {
    'default': RunnableConfig(
        configurable={
            'model': 'gpt-4o',
            'temperature': 0.5,
            'max_tokens': None,
            'timeout': None,
        },
    ),
    'classification_config': RunnableConfig(
        tags=['langsmith:nostream'],
        configurable={
            'model': 'gemini-2.0-flash',
            'model_provider': 'google_genai',
            'temperature': 0.7,
            'max_tokens': 5,
            'timeout': None,
        },
    ),
    'extraction_config': RunnableConfig(
        tags=['langsmith:nostream'],
        configurable={
            'model': 'gpt-4o-mini',
            'temperature': 0.0,
            'max_tokens': None,
            'timeout': None,
        }
    ),
    'thought_config': RunnableConfig(
        tags=['langsmith:nostream'],
        configurable={
            'model': 'deepseek-reasoner',
            'temperature': 0.7,
            'max_tokens': None,
            'timeout': None,
        }
    ),
    'agent_config': RunnableConfig(
        tags=['langsmith:nostream'],
        configurable={
            'model': 'gemini-2.0-flash',
            'model_provider': 'google_genai',
            'temperature': 0.7,
            'max_tokens': None,
            'timeout': None,
        },
        recursion_limit=20
    ),
    'requirements_writing_config': RunnableConfig(
        configurable={
            'model': 'gpt-4o',
            'temperature': 0.5,
            'max_tokens': None,
            'timeout': None,
        },
    ),
}
