from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agents.cats_only_system.agent import cats_graph
from agents.aisa.agent import aisa_graph
from schema import AgentInfo

DEFAULT_AGENT = "Cats Only System"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "Cats Only System": Agent(description="Cats chatbot.", graph=cats_graph),
    "Requirements Assistant": Agent(description="Requirements assistant.", graph=aisa_graph),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
