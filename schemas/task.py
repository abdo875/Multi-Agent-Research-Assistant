"""Task schemas for AgentForge.

Defines the structured decomposition of user requests into discrete,
assigned tasks for specialized worker sub-agents.
"""

from enum import Enum
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CRITIC = "critic"
    SYNTHESIS = "synthesis"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchTask(BaseModel):
    """A bounded task assigned to a specialized sub-agent."""

    task_id: str = Field(description="Unique identifier for the task, e.g. task_research_01")
    task_type: TaskType = Field(description="Type of task to perform")
    assigned_agent: str = Field(description="Name of the assigned agent (e.g. Researcher, Analyst)")
    description: str = Field(description="Clear, actionable task description")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current execution status")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters or focus areas for the task",
    )


class ExecutionPlan(BaseModel):
    """The Supervisor's structured execution plan for fulfilling the user query."""

    objective: str = Field(description="High-level understanding of user objective")
    required_agents: List[str] = Field(description="List of agents required for execution")
    tasks: List[ResearchTask] = Field(description="Discrete tasks to be delegated")
    execution_strategy: str = Field(
        default="parallel_research_and_analysis",
        description="Coordination strategy, e.g. parallel, sequential, or revision",
    )
    rationale: str = Field(
        description="Reasoning behind agent selection and task decomposition"
    )
