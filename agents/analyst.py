"""Analyst Sub-agent for AgentForge.

Specialized worker agent responsible for deep comparative analysis, trade-off evaluation,
latency/cost trade-offs, and practical operational recommendations.
"""

import logging
from typing import List, Optional, Tuple
from core.config import Settings
from core.llm import MockStructuredEngine, generate_structured_output
from schemas.events import EventType, ExecutionEvent
from schemas.responses import AnalysisResult, ResearchResult
from schemas.task import ResearchTask

logger = logging.getLogger("agentforge.analyst")

ANALYST_SYSTEM_PROMPT = """You are the specialized Analyst Sub-agent in a multi-agent system.
Your responsibility:
1. Conduct deep comparative analysis and trade-off evaluation.
2. Evaluate alternatives across multiple engineering dimensions (e.g. latency, cost, freshness, complexity).
3. Synthesize strategic insights and outline real-world practical applications.
4. Return a structured AnalysisResult.
"""


class AnalystAgent:
    """Worker sub-agent specializing in comparative analysis and engineering trade-offs."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def execute(
        self,
        task: ResearchTask,
        query: str,
        research_result: Optional[ResearchResult] = None,
    ) -> Tuple[AnalysisResult, List[ExecutionEvent]]:
        """Execute comparative analysis task and return structured output."""
        start_event = ExecutionEvent(
            agent_name="Analyst",
            event_type=EventType.TASK_STARTED,
            task_description=task.description,
            status="running",
            summary=f"Commenced analysis task '{task.task_id}'. Evaluating dimensions and trade-offs.",
            details={"task_id": task.task_id, "query": query},
        )

        research_context = ""
        if research_result and research_result.findings:
            summarized_findings = [f"{f.topic}: {'; '.join(f.facts[:2])}" for f in research_result.findings]
            research_context = f"\nPre-existing Research Context:\n" + "\n".join(summarized_findings)

        prompt = (
            f"Assigned Analysis Task: {task.description}\n"
            f"Query Topic: {query}\n"
            f"{research_context}\n\n"
            "Produce an AnalysisResult containing:\n"
            "1. Strategic insights.\n"
            "2. Dimension comparisons (Latency, Cost, Data Freshness, Operational Complexity).\n"
            "3. Trade-offs summary.\n"
            "4. Practical industry applications."
        )

        result = generate_structured_output(
            schema=AnalysisResult,
            prompt=prompt,
            settings=self.settings,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            mock_generator_fn=lambda: MockStructuredEngine.generate_analysis(query, research_result),
        )

        completed_event = ExecutionEvent(
            agent_name="Analyst",
            event_type=EventType.TASK_COMPLETED,
            task_description=task.description,
            status="completed",
            summary=f"Completed analysis across {len(result.comparisons)} dimensions with {len(result.insights)} key insights.",
            details={
                "task_id": task.task_id,
                "dimension_count": len(result.comparisons),
                "insight_count": len(result.insights),
            },
        )

        return result, [start_event, completed_event]
