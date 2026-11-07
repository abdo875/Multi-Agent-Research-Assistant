"""End-to-end integration tests for the AgentForge LangGraph workflow.

Tests multi-agent coordination, parallel worker execution, conditional
critic routing, maximum revision safety limits, and centralized state transitions.
"""

import pytest
from core.config import AppMode, Settings
from graph.state import AgentForgeState
from graph.workflow import create_agentforge_graph
from schemas.events import EventType


@pytest.fixture
def base_state():
    return {
        "user_query": "Compare RAG and Fine-Tuning for a customer support application.",
        "execution_plan": None,
        "assigned_tasks": [],
        "research_result": None,
        "analysis_result": None,
        "critic_review": None,
        "revision_count": 0,
        "final_report": None,
        "execution_events": [],
        "errors": [],
        "force_revision_test": False,
    }


def test_full_workflow_execution_mock_mode(base_state):
    """Test full end-to-end execution without API credentials in Mock Mode."""
    graph = create_agentforge_graph()
    output = graph.invoke(base_state)

    assert output["execution_plan"] is not None
    assert len(output["assigned_tasks"]) >= 2
    assert output["research_result"] is not None
    assert output["analysis_result"] is not None
    assert output["critic_review"] is not None
    assert output["final_report"] is not None

    # Check report contents
    report = output["final_report"]
    assert "RAG" in report.title or "Fine-Tuning" in report.title
    assert len(report.key_findings) > 0
    assert len(report.recommendations) > 0

    # Ensure events were logged
    events = output["execution_events"]
    assert len(events) >= 8

    agent_names = {e.agent_name for e in events}
    assert "Supervisor" in agent_names
    assert "Researcher" in agent_names
    assert "Analyst" in agent_names
    assert "Critic" in agent_names
    assert "Synthesizer" in agent_names


def test_parallel_worker_execution(base_state):
    """Verify Researcher and Analyst both execute after Supervisor and before Critic."""
    graph = create_agentforge_graph()
    output = graph.invoke(base_state)
    events = output["execution_events"]

    supervisor_plan_idx = next(
        i for i, e in enumerate(events) if e.agent_name == "Supervisor" and e.event_type == EventType.PLAN_CREATED
    )
    researcher_started_idx = next(
        i for i, e in enumerate(events) if e.agent_name == "Researcher" and e.event_type == EventType.TASK_STARTED
    )
    analyst_started_idx = next(
        i for i, e in enumerate(events) if e.agent_name == "Analyst" and e.event_type == EventType.TASK_STARTED
    )
    critic_idx = next(
        i for i, e in enumerate(events) if e.agent_name == "Critic" and e.event_type == EventType.EVALUATION
    )

    # Both workers must start after supervisor creates the plan
    assert researcher_started_idx > supervisor_plan_idx
    assert analyst_started_idx > supervisor_plan_idx

    # Both workers must be executed before critic evaluates
    assert researcher_started_idx < critic_idx
    assert analyst_started_idx < critic_idx


def test_critic_revision_routing_and_recovery(base_state):
    """Verify conditional edge routes to revision node and loops back to Critic."""
    base_state["force_revision_test"] = True
    graph = create_agentforge_graph()
    output = graph.invoke(base_state)

    assert output["revision_count"] == 1
    events = output["execution_events"]

    # Verify revision event happened
    revision_events = [e for e in events if "Revision cycle" in e.summary]
    assert len(revision_events) == 1

    # Verify Critic evaluated twice: once rejecting, once approving after revision
    critic_events = [e for e in events if e.agent_name == "Critic"]
    assert len(critic_events) == 2
    assert "Needs Revision" in critic_events[0].summary
    assert "Approved" in critic_events[1].summary

    # Ensure Synthesizer still produced the final report
    assert output["final_report"] is not None


def test_maximum_revision_limit_prevents_infinite_loop(base_state):
    """Verify workflow terminates safely even if revisions are repeatedly triggered."""
    # Force revision, but set initial revision_count to 1 (max allowed is 1)
    base_state["force_revision_test"] = True
    base_state["revision_count"] = 1  # already at maximum

    graph = create_agentforge_graph()
    output = graph.invoke(base_state)

    # Must NOT increment beyond 1 because router cuts to synthesizer
    assert output["revision_count"] == 1
    assert output["final_report"] is not None
