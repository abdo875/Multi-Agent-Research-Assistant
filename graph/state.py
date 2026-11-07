"""State definition for AgentForge LangGraph workflow.

Maintains centralized state containing the user query, execution plan,
assigned tasks, agent outputs, critic reviews, revision iterations,
and an append-only timeline of execution events.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from schemas.events import ExecutionEvent
from schemas.responses import (
    AnalysisResult,
    CriticReview,
    ResearchReport,
    ResearchResult,
)
from schemas.task import ExecutionPlan, ResearchTask


class AgentForgeState(TypedDict):
    """Centralized shared state across all LangGraph nodes."""

    user_query: str
    execution_plan: Optional[ExecutionPlan]
    assigned_tasks: List[ResearchTask]
    research_result: Optional[ResearchResult]
    analysis_result: Optional[AnalysisResult]
    critic_review: Optional[CriticReview]
    revision_count: int
    final_report: Optional[ResearchReport]
    execution_events: Annotated[List[ExecutionEvent], operator.add]
    errors: List[str]
    force_revision_test: bool
