"""Execution event schema for observability and audit trail.

Tracks real-time transitions, delegation decisions, parallel execution markers,
critic reviews, and handoffs across the multi-agent graph.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class EventType(str, Enum):
    PLAN_CREATED = "plan_created"
    TASK_DELEGATED = "task_delegated"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    EVALUATION = "evaluation"
    ROUTING = "routing"
    REVISION_REQUESTED = "revision_requested"
    REPORT_GENERATED = "report_generated"
    ERROR = "error"


def default_timestamp() -> str:
    """Generate ISO timestamp in UTC."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


class ExecutionEvent(BaseModel):
    """Structured record of a single agent or orchestration event."""

    timestamp: str = Field(default_factory=default_timestamp, description="Timestamp of the event")
    agent_name: str = Field(description="Name of agent producing the event (Supervisor, Researcher, etc.)")
    event_type: EventType = Field(description="Classification of the event")
    task_description: str = Field(description="Brief description of the task or action")
    status: str = Field(default="completed", description="Execution status: pending, running, completed, error")
    summary: str = Field(description="Concise human-readable summary of the action or decision")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Optional extra metadata or metrics")
