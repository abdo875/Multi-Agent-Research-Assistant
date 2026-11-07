"""LangGraph workflow node implementations for AgentForge.

Each node represents an autonomous agent execution step, reading from and
writing to the centralized AgentForgeState.
"""

import logging
from typing import Any, Dict
from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.researcher import ResearcherAgent
from agents.supervisor import SupervisorAgent
from agents.synthesizer import SynthesizerAgent
from core.config import get_settings
from graph.state import AgentForgeState
from schemas.events import EventType, ExecutionEvent
from schemas.task import ResearchTask, TaskStatus, TaskType

logger = logging.getLogger("agentforge.nodes")


def supervisor_node(state: AgentForgeState) -> Dict[str, Any]:
    """Supervisor orchestrates task decomposition and delegation."""
    settings = get_settings()
    supervisor = SupervisorAgent(settings)
    query = state["user_query"]

    plan, events = supervisor.create_plan(query)

    return {
        "execution_plan": plan,
        "assigned_tasks": plan.tasks,
        "execution_events": events,
    }


def researcher_node(state: AgentForgeState) -> Dict[str, Any]:
    """Researcher sub-agent executes assigned factual retrieval task."""
    settings = get_settings()
    researcher = ResearcherAgent(settings)
    query = state["user_query"]

    # Locate research task from state or fallback
    assigned_tasks = state.get("assigned_tasks") or []
    task = next(
        (t for t in assigned_tasks if t.assigned_agent.lower() == "researcher" or t.task_type == TaskType.RESEARCH),
        ResearchTask(
            task_id="task_research_auto",
            task_type=TaskType.RESEARCH,
            assigned_agent="Researcher",
            description=f"Conduct factual research on: {query}",
            status=TaskStatus.IN_PROGRESS,
        ),
    )

    result, events = researcher.execute(task, query)

    return {
        "research_result": result,
        "execution_events": events,
    }


def analyst_node(state: AgentForgeState) -> Dict[str, Any]:
    """Analyst sub-agent executes assigned trade-off evaluation task."""
    settings = get_settings()
    analyst = AnalystAgent(settings)
    query = state["user_query"]

    # Locate analysis task from state or fallback
    assigned_tasks = state.get("assigned_tasks") or []
    task = next(
        (t for t in assigned_tasks if t.assigned_agent.lower() == "analyst" or t.task_type == TaskType.ANALYSIS),
        ResearchTask(
            task_id="task_analysis_auto",
            task_type=TaskType.ANALYSIS,
            assigned_agent="Analyst",
            description=f"Analyze trade-offs and operational metrics for: {query}",
            status=TaskStatus.IN_PROGRESS,
        ),
    )

    # Note: in parallel execution, research_result might complete before or concurrently
    research_result = state.get("research_result")
    result, events = analyst.execute(task, query, research_result)

    return {
        "analysis_result": result,
        "execution_events": events,
    }


def critic_node(state: AgentForgeState) -> Dict[str, Any]:
    """Critic sub-agent reviews deliverables against the original query."""
    settings = get_settings()
    critic = CriticAgent(settings)
    query = state["user_query"]
    research = state.get("research_result")
    analysis = state.get("analysis_result")
    revision_count = state.get("revision_count", 0)
    force_rev = state.get("force_revision_test", False)

    review, events = critic.evaluate(
        query=query,
        research_result=research,
        analysis_result=analysis,
        revision_count=revision_count,
        force_revision_test=force_rev,
    )

    return {
        "critic_review": review,
        "execution_events": events,
    }


def revision_node(state: AgentForgeState) -> Dict[str, Any]:
    """Revision node triggers targeted refinements based on Critic feedback."""
    revision_count = state.get("revision_count", 0) + 1
    critic_review = state.get("critic_review")

    feedback = (
        critic_review.revision_instructions
        if critic_review and critic_review.revision_instructions
        else "Deepen technical comparisons and operational constraints."
    )

    event = ExecutionEvent(
        agent_name="Supervisor",
        event_type=EventType.TASK_STARTED,
        task_description="Targeted revision and deliverable enrichment",
        status="completed",
        summary=f"Revision cycle {revision_count} enacted. Incorporating critic instructions: {feedback[:100]}...",
        details={
            "revision_count": revision_count,
            "instructions": feedback,
        },
    )

    # We also enrich the analysis with the revised insights
    analysis = state.get("analysis_result")
    if analysis:
        # Augment insights with the critic instructions addressed
        new_insights = list(analysis.insights)
        new_insights.append(f"[Revision {revision_count} Added]: Addressed critic mandate regarding '{feedback}'.")
        analysis.insights = new_insights

    return {
        "revision_count": revision_count,
        "analysis_result": analysis,
        "execution_events": [event],
    }


def synthesizer_node(state: AgentForgeState) -> Dict[str, Any]:
    """Synthesizer sub-agent consolidates findings into final report."""
    settings = get_settings()
    synthesizer = SynthesizerAgent(settings)
    query = state["user_query"]
    research = state.get("research_result")
    analysis = state.get("analysis_result")
    critic = state.get("critic_review")
    revision_count = state.get("revision_count", 0)

    report, events = synthesizer.execute(
        query=query,
        research_result=research,
        analysis_result=analysis,
        critic_review=critic,
        metadata={
            "revision_count": revision_count,
            "tasks_executed": len(state.get("assigned_tasks", [])),
        },
    )

    return {
        "final_report": report,
        "execution_events": events,
    }
