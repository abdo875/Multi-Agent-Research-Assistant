"""Synthesizer Sub-agent for AgentForge.

Consolidates approved outputs from Researcher, Analyst, and Critic into
a comprehensive, executive-ready technical research report.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from core.config import Settings
from core.llm import MockStructuredEngine, generate_structured_output
from schemas.events import EventType, ExecutionEvent
from schemas.responses import (
    AnalysisResult,
    CriticReview,
    ResearchReport,
    ResearchResult,
)

logger = logging.getLogger("agentforge.synthesizer")

SYNTHESIZER_SYSTEM_PROMPT = """You are the specialized Synthesizer Sub-agent in a multi-agent system.
Your responsibility:
1. Merge the approved deliverables from Researcher and Analyst into a polished research report.
2. Formulate a compelling Executive Summary.
3. Highlight Key Findings, Comparative Analysis, Trade-offs, Recommendations, and Limitations.
4. Eliminate redundant text and maintain an authoritative, technical engineering tone.
5. Return a structured ResearchReport.
"""


class SynthesizerAgent:
    """Worker sub-agent specializing in final synthesis and report generation."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def execute(
        self,
        query: str,
        research_result: Optional[ResearchResult],
        analysis_result: Optional[AnalysisResult],
        critic_review: Optional[CriticReview],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ResearchReport, List[ExecutionEvent]]:
        """Synthesize approved deliverables into a final structured research report."""
        prompt = (
            f"Original Query: {query}\n"
            f"Research Deliverable:\n{research_result.model_dump_json(indent=2) if research_result else 'None'}\n\n"
            f"Analysis Deliverable:\n{analysis_result.model_dump_json(indent=2) if analysis_result else 'None'}\n\n"
            f"Critic Evaluation:\n{critic_review.model_dump_json(indent=2) if critic_review else 'None'}\n\n"
            "Synthesize these inputs into a final, comprehensive ResearchReport."
        )

        meta = {
            "app_mode": self.settings.app_mode.value,
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model,
            **(metadata or {}),
        }

        report = generate_structured_output(
            schema=ResearchReport,
            prompt=prompt,
            settings=self.settings,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            mock_generator_fn=lambda: MockStructuredEngine.generate_report(
                query,
                research_result,
                analysis_result,
                critic_review,
                metadata=meta,
            ),
        )

        event = ExecutionEvent(
            agent_name="Synthesizer",
            event_type=EventType.REPORT_GENERATED,
            task_description="Synthesis and final technical report compilation",
            status="completed",
            summary=f"Compiled final research report: '{report.title}' with {len(report.recommendations)} strategic recommendations.",
            details={
                "report_title": report.title,
                "key_findings_count": len(report.key_findings),
                "trade_offs_count": len(report.trade_offs),
                "recommendations_count": len(report.recommendations),
            },
        )

        return report, [event]
