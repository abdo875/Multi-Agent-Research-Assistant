"""Supervisor Agent for AgentForge.

The central orchestrator in the Supervisor-Worker multi-agent architecture.
Responsible for task decomposition, agent delegation, tracking execution,
and conditional workflow routing based on Critic feedback.
"""

import logging
from typing import List, Tuple
from core.config import Settings
from core.llm import MockStructuredEngine, generate_structured_output
from schemas.events import EventType, ExecutionEvent
from schemas.responses import CriticReview
from schemas.task import ExecutionPlan, ResearchTask

logger = logging.getLogger("agentforge.supervisor")

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Agent in a specialized multi-agent research assistant system.
Your responsibility is NOT to answer the research question directly, but to orchestrate worker agents.
You must:
1. Understand the user's objective and core technical question.
2. Formulate a structured execution plan.
3. Decompose the request into independent, specialized sub-tasks for the 'Researcher' and 'Analyst' worker sub-agents.
4. Specify clear task descriptions and target parameters.
"""


class SupervisorAgent:
    """Central orchestrating supervisor coordinating sub-agent workers."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def create_plan(self, query: str) -> Tuple[ExecutionPlan, List[ExecutionEvent]]:
        """Decompose user query into an ExecutionPlan and emit initial delegation events."""
        prompt = (
            f"User Research Query: {query}\n\n"
            "Create an ExecutionPlan decomposing this inquiry into two concurrent tasks:\n"
            "1. Task for 'Researcher': Factual specifications, architectures, mechanisms, and examples.\n"
            "2. Task for 'Analyst': Comparative trade-offs, operational implications, latency/cost analysis, and practical recommendations.\n"
            "Return the structured ExecutionPlan."
        )

        plan = generate_structured_output(
            schema=ExecutionPlan,
            prompt=prompt,
            settings=self.settings,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            mock_generator_fn=lambda: MockStructuredEngine.generate_plan(query),
        )

        events: List[ExecutionEvent] = [
            ExecutionEvent(
                agent_name="Supervisor",
                event_type=EventType.PLAN_CREATED,
                task_description="Objective analysis and task decomposition",
                status="completed",
                summary=f"Formulated execution plan with {len(plan.tasks)} delegated tasks ({plan.execution_strategy}).",
                details={
                    "required_agents": plan.required_agents,
                    "task_count": len(plan.tasks),
                    "rationale": plan.rationale,
                },
            )
        ]

        # Emit delegation events for each task in the plan
        for task in plan.tasks:
            events.append(
                ExecutionEvent(
                    agent_name="Supervisor",
                    event_type=EventType.TASK_DELEGATED,
                    task_description=f"Delegation to {task.assigned_agent}",
                    status="completed",
                    summary=f"Delegated '{task.task_type.value}' task ({task.task_id}) to {task.assigned_agent}.",
                    details={
                        "task_id": task.task_id,
                        "assigned_agent": task.assigned_agent,
                        "description": task.description,
                    },
                )
            )

        return plan, events

    def route_critic_decision(
        self,
        critic_review: CriticReview,
        revision_count: int,
        max_revisions: int,
    ) -> Tuple[str, ExecutionEvent]:
        """Determine whether to route workflow to revision or directly to synthesis."""
        needs_revision = critic_review.needs_revision
        can_revise = revision_count < max_revisions

        if needs_revision and can_revise:
            route = "revision"
            summary = (
                f"Critic identified gaps (score: {critic_review.completeness_score:.2f}). "
                f"Routing to revision cycle ({revision_count + 1}/{max_revisions})."
            )
            event_type = EventType.REVISION_REQUESTED
        else:
            route = "synthesizer"
            reason = "Approved by Critic" if not needs_revision else f"Max revision limit reached ({max_revisions})"
            summary = f"{reason} (score: {critic_review.completeness_score:.2f}). Routing to Synthesizer."
            event_type = EventType.ROUTING

        event = ExecutionEvent(
            agent_name="Supervisor",
            event_type=event_type,
            task_description="Conditional workflow routing evaluation",
            status="completed",
            summary=summary,
            details={
                "target_node": route,
                "revision_count": revision_count,
                "max_revisions": max_revisions,
                "needs_revision": needs_revision,
                "completeness_score": critic_review.completeness_score,
            },
        )

        return route, event
