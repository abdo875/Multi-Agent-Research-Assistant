"""Unit tests for individual AgentForge specialized sub-agents.

Verifies task planning, delegation, research retrieval, comparative analysis,
critic evaluation, and final report synthesis with Pydantic validation.
"""

import pytest
from agents.analyst import AnalystAgent
from agents.critic import CriticAgent
from agents.researcher import ResearcherAgent
from agents.supervisor import SupervisorAgent
from agents.synthesizer import SynthesizerAgent
from core.config import AppMode, Settings
from schemas.events import EventType
from schemas.responses import (
    AnalysisResult,
    CriticReview,
    ResearchReport,
    ResearchResult,
)
from schemas.task import ExecutionPlan, TaskStatus, TaskType


@pytest.fixture
def mock_settings():
    return Settings(app_mode=AppMode.MOCK, max_revision_attempts=1)


def test_supervisor_task_planning(mock_settings):
    """Verify Supervisor generates a structured ExecutionPlan with assigned worker tasks."""
    supervisor = SupervisorAgent(mock_settings)
    query = "Compare RAG and Fine-Tuning for a customer support application."

    plan, events = supervisor.create_plan(query)

    assert isinstance(plan, ExecutionPlan)
    assert len(plan.tasks) >= 2
    assert "Researcher" in plan.required_agents
    assert "Analyst" in plan.required_agents

    # Verify delegation events
    plan_events = [e for e in events if e.event_type == EventType.PLAN_CREATED]
    assert len(plan_events) == 1
    delegation_events = [e for e in events if e.event_type == EventType.TASK_DELEGATED]
    assert len(delegation_events) == len(plan.tasks)


def test_task_delegation_attributes(mock_settings):
    """Verify each task has valid status, IDs, and bounded scope."""
    supervisor = SupervisorAgent(mock_settings)
    plan, _ = supervisor.create_plan("Vector DB selection: Qdrant vs Milvus")

    for task in plan.tasks:
        assert task.task_id.startswith("task_")
        assert task.status == TaskStatus.PENDING
        assert task.task_type in [TaskType.RESEARCH, TaskType.ANALYSIS]
        assert task.assigned_agent in ["Researcher", "Analyst"]


def test_researcher_execution_and_tool_grounding(mock_settings):
    """Verify Researcher executes task using knowledge retrieval and outputs ResearchResult."""
    supervisor = SupervisorAgent(mock_settings)
    plan, _ = supervisor.create_plan("Compare RAG and Fine-Tuning")
    research_task = [t for t in plan.tasks if t.task_type == TaskType.RESEARCH][0]

    researcher = ResearcherAgent(mock_settings)
    result, events = researcher.execute(research_task, "Compare RAG and Fine-Tuning")

    assert isinstance(result, ResearchResult)
    assert len(result.findings) >= 1
    assert len(result.sources_consulted) >= 1
    assert any(f.source_type == "retrieved_reference" for f in result.findings)

    # Check event sequence
    assert events[0].event_type == EventType.TASK_STARTED
    assert events[1].event_type == EventType.TASK_COMPLETED


def test_analyst_comparative_dimensions(mock_settings):
    """Verify Analyst evaluates multidimensional trade-offs and returns AnalysisResult."""
    supervisor = SupervisorAgent(mock_settings)
    plan, _ = supervisor.create_plan("Compare RAG and Fine-Tuning")
    analysis_task = [t for t in plan.tasks if t.task_type == TaskType.ANALYSIS][0]

    analyst = AnalystAgent(mock_settings)
    result, events = analyst.execute(analysis_task, "Compare RAG and Fine-Tuning")

    assert isinstance(result, AnalysisResult)
    assert len(result.comparisons) >= 2
    assert len(result.insights) >= 1
    assert len(result.practical_applications) >= 1

    dimensions = [c.dimension for c in result.comparisons]
    assert any("Latency" in d or "Compute" in d for d in dimensions)
    assert any("Freshness" in d or "Data" in d for d in dimensions)


def test_critic_approval(mock_settings):
    """Verify Critic outputs an approval review when criteria are satisfied."""
    critic = CriticAgent(mock_settings)
    query = "Compare RAG and Fine-Tuning"

    researcher = ResearcherAgent(mock_settings)
    supervisor = SupervisorAgent(mock_settings)
    plan, _ = supervisor.create_plan(query)
    res_task = [t for t in plan.tasks if t.task_type == TaskType.RESEARCH][0]
    ana_task = [t for t in plan.tasks if t.task_type == TaskType.ANALYSIS][0]

    research, _ = researcher.execute(res_task, query)
    analyst = AnalystAgent(mock_settings)
    analysis, _ = analyst.execute(ana_task, query, research)

    review, events = critic.evaluate(query, research, analysis, revision_count=0, force_revision_test=False)

    assert isinstance(review, CriticReview)
    assert review.objective_satisfied is True
    assert review.completeness_score >= 0.8
    assert review.needs_revision is False
    assert events[0].event_type == EventType.EVALUATION


def test_critic_revision_mandate(mock_settings):
    """Verify Critic flags revision when gaps are intentionally simulated."""
    critic = CriticAgent(mock_settings)
    query = "Compare RAG and Fine-Tuning"

    review, events = critic.evaluate(
        query=query,
        research_result=None,
        analysis_result=None,
        revision_count=0,
        force_revision_test=True,
    )

    assert isinstance(review, CriticReview)
    assert review.needs_revision is True
    assert review.completeness_score < 0.8
    assert review.revision_instructions is not None


def test_synthesizer_report_generation(mock_settings):
    """Verify Synthesizer consolidates inputs into a complete ResearchReport."""
    synthesizer = SynthesizerAgent(mock_settings)
    query = "Compare RAG and Fine-Tuning"

    researcher = ResearcherAgent(mock_settings)
    supervisor = SupervisorAgent(mock_settings)
    plan, _ = supervisor.create_plan(query)
    res_task = [t for t in plan.tasks if t.task_type == TaskType.RESEARCH][0]
    ana_task = [t for t in plan.tasks if t.task_type == TaskType.ANALYSIS][0]

    research, _ = researcher.execute(res_task, query)
    analyst = AnalystAgent(mock_settings)
    analysis, _ = analyst.execute(ana_task, query, research)
    critic = CriticAgent(mock_settings)
    review, _ = critic.evaluate(query, research, analysis, revision_count=0)

    report, events = synthesizer.execute(query, research, analysis, review)

    assert isinstance(report, ResearchReport)
    assert len(report.title) > 0
    assert len(report.executive_summary) > 50
    assert len(report.key_findings) >= 1
    assert len(report.recommendations) >= 2
    assert len(report.limitations) >= 1
    assert events[0].event_type == EventType.REPORT_GENERATED
