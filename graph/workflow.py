"""LangGraph StateGraph orchestration for AgentForge.

Implements the multi-agent graph with parallel worker execution,
centralized state transitions, and conditional revision loops.
"""

from langgraph.graph import END, START, StateGraph
from core.config import get_settings
from graph.nodes import (
    analyst_node,
    critic_node,
    researcher_node,
    revision_node,
    supervisor_node,
    synthesizer_node,
)
from graph.state import AgentForgeState


def route_after_critic(state: AgentForgeState) -> str:
    """Evaluate Critic review and decide whether to loop for revision or proceed to synthesis."""
    settings = get_settings()
    review = state.get("critic_review")
    revision_count = state.get("revision_count", 0)
    max_revisions = settings.max_revision_attempts

    if review and review.needs_revision and revision_count < max_revisions:
        return "revision"
    return "synthesizer"


def create_agentforge_graph() -> StateGraph:
    """Build and compile the AgentForge LangGraph workflow.

    Workflow topology:
        START
          ↓
      Supervisor
       ↙      ↘   (Parallel Execution)
    Researcher  Analyst
       ↘      ↙   (Synchronized Fan-in)
        Critic
          ↓
     [Conditional]
      ↙         ↘
   Revision     Synthesizer
      ↓            ↓
    Critic        END
    """
    workflow = StateGraph(AgentForgeState)

    # Register agent nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("revision", revision_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Initial delegation edge
    workflow.add_edge(START, "supervisor")

    # Parallel worker execution: Fan-out from supervisor to researcher & analyst
    workflow.add_edge("supervisor", "researcher")
    workflow.add_edge("supervisor", "analyst")

    # Synchronized fan-in: both workers deliver to the critic
    workflow.add_edge("researcher", "critic")
    workflow.add_edge("analyst", "critic")

    # Conditional edge: Critic evaluation directs to Revision or Synthesizer
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "revision": "revision",
            "synthesizer": "synthesizer",
        },
    )

    # Revision cycle loops back to Critic for re-evaluation
    workflow.add_edge("revision", "critic")

    # Synthesizer marks the end of execution
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
