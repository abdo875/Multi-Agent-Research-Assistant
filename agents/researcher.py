"""Researcher Sub-agent for AgentForge.

Specialized worker agent responsible for factual retrieval, technical specification
discovery, architectural definitions, and grounded examples.
"""

import logging
from typing import List, Tuple
from core.config import Settings
from core.llm import MockStructuredEngine, generate_structured_output
from schemas.events import EventType, ExecutionEvent
from schemas.responses import ResearchResult
from schemas.task import ResearchTask
from tools.research_tools import query_research_knowledge

logger = logging.getLogger("agentforge.researcher")

RESEARCHER_SYSTEM_PROMPT = """You are the specialized Researcher Sub-agent in a multi-agent system.
Your responsibility:
1. Conduct factual technical research on the assigned topic.
2. Identify architectural specifications, mechanics, components, and grounded examples.
3. Clearly distinguish retrieved reference facts from parametric knowledge.
4. Return a structured ResearchResult.
"""


class ResearcherAgent:
    """Worker sub-agent specializing in technical facts and architectural retrieval."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def execute(self, task: ResearchTask, query: str) -> Tuple[ResearchResult, List[ExecutionEvent]]:
        """Execute research task and return structured results with observability events."""
        start_event = ExecutionEvent(
            agent_name="Researcher",
            event_type=EventType.TASK_STARTED,
            task_description=task.description,
            status="running",
            summary=f"Commenced research task '{task.task_id}'. Consulting domain knowledge tool.",
            details={"task_id": task.task_id, "query": query},
        )

        # Retrieve factual grounding from tool
        tool_data = query_research_knowledge(query)

        prompt = (
            f"Assigned Research Task: {task.description}\n"
            f"Query Topic: {query}\n"
            f"Reference Grounding Retrieved:\n"
            f"- Facts: {tool_data['facts']}\n"
            f"- Technical Details: {tool_data['technical_details']}\n"
            f"- Examples: {tool_data['examples']}\n\n"
            "Format the findings into a structured ResearchResult with individual ResearchFinding items."
        )

        result = generate_structured_output(
            schema=ResearchResult,
            prompt=prompt,
            settings=self.settings,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            mock_generator_fn=lambda: MockStructuredEngine.generate_research(query),
        )

        completed_event = ExecutionEvent(
            agent_name="Researcher",
            event_type=EventType.TASK_COMPLETED,
            task_description=task.description,
            status="completed",
            summary=f"Completed research with {len(result.findings)} key finding(s) from {len(result.sources_consulted)} source(s).",
            details={
                "task_id": task.task_id,
                "findings_count": len(result.findings),
                "sources": result.sources_consulted,
            },
        )

        return result, [start_event, completed_event]
