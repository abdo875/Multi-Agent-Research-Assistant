"""Critic / Reviewer Sub-agent for AgentForge.

Quality control agent in the workflow. Evaluates research and analysis against
the original objective, checks for contradictions and gaps, and issues
structured approval or revision mandates.
"""

import logging
from typing import List, Optional, Tuple
from core.config import Settings
from core.llm import MockStructuredEngine, generate_structured_output
from schemas.events import EventType, ExecutionEvent
from schemas.responses import AnalysisResult, CriticReview, ResearchResult

logger = logging.getLogger("agentforge.critic")

CRITIC_SYSTEM_PROMPT = """You are the specialized Critic / Reviewer Sub-agent in a multi-agent system.
Your responsibility:
1. Objectively evaluate the combined research and analysis outputs.
2. Verify whether the user's research objective is directly and thoroughly addressed.
3. Detect missing dimensions, weak arguments, or contradictions.
4. Score completeness from 0.0 to 1.0.
5. Determine if revision is strictly required (needs_revision = True) or if quality is acceptable for synthesis.
"""


class CriticAgent:
    """Quality assurance and critique sub-agent."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def evaluate(
        self,
        query: str,
        research_result: Optional[ResearchResult],
        analysis_result: Optional[AnalysisResult],
        revision_count: int = 0,
        force_revision_test: bool = False,
    ) -> Tuple[CriticReview, List[ExecutionEvent]]:
        """Evaluate agent deliverables against the core query."""
        research_summary = research_result.summary if research_result else "No research available"
        analysis_summary = analysis_result.trade_offs_summary if analysis_result else "No analysis available"

        prompt = (
            f"Original User Query: {query}\n"
            f"Current Revision Cycle: {revision_count}\n"
            f"Researcher Deliverable: {research_summary}\n"
            f"Analyst Deliverable: {analysis_summary}\n\n"
            "Perform a rigorous technical critique:\n"
            "- Did the workers address the query directly?\n"
            "- Are there any omitted critical dimensions?\n"
            "- Are there contradictions?\n"
            "- Assign a completeness_score (0.0 to 1.0).\n"
            "- Set needs_revision to True only if critical gaps exist, else False."
        )

        review = generate_structured_output(
            schema=CriticReview,
            prompt=prompt,
            settings=self.settings,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            mock_generator_fn=lambda: MockStructuredEngine.generate_critic(
                query,
                research_result,
                analysis_result,
                revision_count=revision_count,
                force_revision_test=force_revision_test,
            ),
        )

        status_label = "Needs Revision" if review.needs_revision else "Approved"
        summary = (
            f"Critique outcome: {status_label} (Completeness: {review.completeness_score:.0%}). "
            f"{review.quality_assessment[:100]}..."
        )

        event = ExecutionEvent(
            agent_name="Critic",
            event_type=EventType.EVALUATION,
            task_description="Quality evaluation and completeness audit",
            status="completed",
            summary=summary,
            details={
                "objective_satisfied": review.objective_satisfied,
                "completeness_score": review.completeness_score,
                "needs_revision": review.needs_revision,
                "missing_information": review.missing_information,
                "contradictions": review.contradictions_detected,
            },
        )

        return review, [event]
