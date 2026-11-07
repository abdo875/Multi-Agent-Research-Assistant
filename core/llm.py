"""LLM Integration Layer for AgentForge.

Bridges LangChain, OpenAI-compatible APIs (including local vLLM), and Mock Mode.
Provides structured output generation with graceful fallbacks.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Type, TypeVar
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from core.config import AppMode, Settings
from schemas.events import EventType, ExecutionEvent
from schemas.responses import (
    AnalysisDimension,
    AnalysisResult,
    CriticReview,
    ResearchFinding,
    ResearchReport,
    ResearchResult,
)
from schemas.task import ExecutionPlan, ResearchTask, TaskStatus, TaskType
from tools.research_tools import query_research_knowledge

logger = logging.getLogger("agentforge.llm")

T = TypeVar("T", bound=BaseModel)


def get_chat_model(settings: Settings) -> Optional[ChatOpenAI]:
    """Instantiate a LangChain ChatOpenAI client if not in Mock Mode.

    Supports both remote OpenAI-compatible providers and local vLLM instances.
    """
    if settings.is_mock():
        return None

    api_key = settings.llm_api_key or "EMPTY"  # vLLM allows dummy/empty key

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        base_url=settings.llm_base_url,
        api_key=api_key,
        max_retries=2,
    )


class MockStructuredEngine:
    """Deterministic structured output generator for interview demonstrations and testing."""

    @staticmethod
    def generate_plan(query: str) -> ExecutionPlan:
        """Create Supervisor execution plan."""
        return ExecutionPlan(
            objective=f"Conduct an in-depth comparative investigation into: '{query}'",
            required_agents=["Researcher", "Analyst", "Critic", "Synthesizer"],
            execution_strategy="parallel_research_and_analysis",
            rationale=(
                "Decomposed into concurrent factual retrieval (Researcher) and trade-off "
                "evaluation (Analyst) to demonstrate multi-agent concurrency, followed by "
                "critic quality gating and final report synthesis."
            ),
            tasks=[
                ResearchTask(
                    task_id="task_research_01",
                    task_type=TaskType.RESEARCH,
                    assigned_agent="Researcher",
                    description=f"Retrieve technical concepts, factual specifications, and architecture details for: {query}",
                    status=TaskStatus.PENDING,
                    parameters={"query": query, "focus": "factual_specifications"},
                ),
                ResearchTask(
                    task_id="task_analysis_01",
                    task_type=TaskType.ANALYSIS,
                    assigned_agent="Analyst",
                    description=f"Analyze engineering trade-offs, operational complexity, and practical applications for: {query}",
                    status=TaskStatus.PENDING,
                    parameters={"query": query, "focus": "comparative_trade_offs"},
                ),
            ],
        )

    @staticmethod
    def generate_research(query: str) -> ResearchResult:
        """Create Researcher structured result."""
        tool_data = query_research_knowledge(query)

        findings: list[ResearchFinding] = []
        if "rag" in query.lower() and "fine" in query.lower():
            findings.append(
                ResearchFinding(
                    topic="Retrieval-Augmented Generation (RAG)",
                    facts=[
                        "Decouples external knowledge storage from model parameters using vector indexes.",
                        "Enables real-time document updates and immediate data deletion without retraining.",
                        "Supports precise source citations and document-level access permissions (ACLs).",
                        "Reduces hallucination rate on domain-specific customer support queries.",
                    ],
                    technical_details=(
                        "Consists of a bi-encoder embedding model, an ANN vector database (e.g. Qdrant/Milvus), "
                        "and a cross-encoder re-ranker. Top-k relevant chunks are dynamically injected into the system context."
                    ),
                    examples=[
                        "Customer service assistant indexing constantly updated product manuals, pricing FAQs, and return policies.",
                        "Multi-tenant customer support with role-based document access controls.",
                    ],
                    source_type="retrieved_reference",
                )
            )
            findings.append(
                ResearchFinding(
                    topic="Supervised Fine-Tuning (SFT / PEFT)",
                    facts=[
                        "Updates internal transformer weights (often via LoRA/QLoRA) using paired conversation data.",
                        "Bakes domain vocabulary, tone of voice, and specialized output structure directly into the model.",
                        "Does not require a vector search lookup during token generation.",
                        "Knowledge is static as of the training run; prone to hallucinations if facts change.",
                    ],
                    technical_details=(
                        "LoRA freezes pretrained weights and injects low-rank trainable matrices (A and B) into attention projections. "
                        "Significantly lowers VRAM requirements while teaching rigid dialogue escalation behavior."
                    ),
                    examples=[
                        "Customer support model trained to strictly adhere to standard operating escalation codes.",
                        "Style adaptation ensuring all customer responses maintain a warm, compliant brand persona.",
                    ],
                    source_type="retrieved_reference",
                )
            )
        else:
            findings.append(
                ResearchFinding(
                    topic=f"Technical Core: {query}",
                    facts=tool_data["facts"][:4],
                    technical_details=tool_data["technical_details"],
                    examples=tool_data["examples"],
                    source_type="retrieved_reference",
                )
            )

        return ResearchResult(
            task_id="task_research_01",
            findings=findings,
            sources_consulted=tool_data["sources"],
            summary=f"Retrieved verifiable architecture details and technical specifications for {query}.",
        )

    @staticmethod
    def generate_analysis(query: str, research: Optional[ResearchResult] = None) -> AnalysisResult:
        """Create Analyst structured result."""
        dimensions = [
            AnalysisDimension(
                dimension="Data Freshness & Updatability",
                comparison=(
                    "RAG updates instantly upon vector index upsert; "
                    "Fine-Tuning requires new training data preparation, validation, and checkpoint deployment."
                ),
                trade_offs="RAG trades index maintenance for zero retraining latency; Fine-Tuning incurs delayed updates.",
            ),
            AnalysisDimension(
                dimension="Latency & Compute Overhead",
                comparison=(
                    "RAG introduces vector search + reranking latency (~50-150ms) plus larger prompt token overhead; "
                    "Fine-Tuned models use shorter prompts with zero retrieval overhead, leading to faster Time-to-First-Token (TTFT)."
                ),
                trade_offs="RAG increases inference token cost; Fine-Tuning incurs upfront training compute but leaner inference.",
            ),
            AnalysisDimension(
                dimension="Hallucination & Verifiability",
                comparison=(
                    "RAG provides direct grounded citations pointing to exact document chunks; "
                    "Fine-Tuned models rely on parametric memory and may hallucinate plausible-sounding falsehoods."
                ),
                trade_offs="RAG guarantees auditable answers; Fine-Tuning provides behavioral consistency but weaker factual auditability.",
            ),
            AnalysisDimension(
                dimension="Operational & Engineering Complexity",
                comparison=(
                    "RAG requires vector database management, chunking pipelines, and embedding drift monitoring; "
                    "Fine-Tuning requires GPU training infrastructure, dataset curation, MLOps evaluation, and versioned weight hosting."
                ),
                trade_offs="RAG operationalizes search infrastructure; Fine-Tuning operationalizes model lifecycle and weights.",
            ),
        ]

        insights = [
            "For customer support applications, dynamic knowledge (policies, pricing, products) mandates a RAG retrieval layer.",
            "Fine-Tuning is best reserved for specialized behavioral adaptation: concise tone, strict JSON schema output, and triage classification.",
            "The industry gold standard is a hybrid architecture: a lightly fine-tuned or prompt-aligned model querying an indexed RAG knowledge base.",
        ]

        practical_applications = [
            "Tier-1 Support Bot: RAG indexing helpdesk knowledge base with real-time policy updates.",
            "Ticket Classifier / Router: Small fine-tuned model for instant, low-latency triage.",
            "Enterprise Hybrid Assistant: LoRA fine-tuned model for tone paired with hybrid vector/keyword search for precision.",
        ]

        return AnalysisResult(
            task_id="task_analysis_01",
            insights=insights,
            comparisons=dimensions,
            trade_offs_summary="RAG dominates for dynamic, factual knowledge; Fine-Tuning excels at behavior and syntax compliance.",
            practical_applications=practical_applications,
        )

    @staticmethod
    def generate_critic(
        query: str,
        research: Optional[ResearchResult],
        analysis: Optional[AnalysisResult],
        revision_count: int = 0,
        force_revision_test: bool = False,
    ) -> CriticReview:
        """Create Critic review."""
        # For demonstration: If force_revision_test is true and revision_count == 0, trigger one revision cycle!
        if force_revision_test and revision_count == 0:
            return CriticReview(
                objective_satisfied=False,
                completeness_score=0.72,
                missing_information=["Explicit cost breakdown per 1M tokens", "SLA comparison for peak customer support hours"],
                contradictions_detected=[],
                quality_assessment="Solid foundational analysis, but requires concrete numerical SLAs and cost comparisons to fully answer customer support production constraints.",
                needs_revision=True,
                revision_instructions="Incorporate cost-per-query analysis and latency SLA comparison for high-concurrency customer support.",
            )

        return CriticReview(
            objective_satisfied=True,
            completeness_score=0.94,
            missing_information=[],
            contradictions_detected=[],
            quality_assessment="Thorough, balanced comparative breakdown addressing factual specifications, operational trade-offs, and architecture decisions.",
            needs_revision=False,
            revision_instructions=None,
        )

    @staticmethod
    def generate_report(
        query: str,
        research: Optional[ResearchResult],
        analysis: Optional[AnalysisResult],
        critic: Optional[CriticReview],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchReport:
        """Create Synthesizer final research report."""
        findings_bullets = []
        if research and research.findings:
            for f in research.findings:
                findings_bullets.append(f"**{f.topic}**: {'; '.join(f.facts[:2])}")
        else:
            findings_bullets.append(f"Grounded technical overview formulated for: {query}")

        trade_offs = []
        if analysis and analysis.comparisons:
            for c in analysis.comparisons:
                trade_offs.append(f"**{c.dimension}**: {c.trade_offs}")
        else:
            trade_offs.append("Balanced trade-offs across cost, latency, accuracy, and engineering complexity.")

        recommendations = [
            "Start with RAG for customer-facing knowledge: Provides auditable citations, zero training latency, and simple knowledge maintenance.",
            "Evaluate Fine-Tuning when tone, format compliance, or latency demands cannot be met via prompt engineering alone.",
            "Adopt a Hybrid Strategy for mature systems: Use PEFT/LoRA to shape response style while fetching dynamic data via RAG.",
            "Deploy an inference engine like vLLM with PagedAttention when serving high-concurrency local open models to reduce KV cache memory pressure.",
        ]

        limitations = [
            "Parametric benchmarks reflect standard 8B-70B model classes and general cloud pricing as of 2025/2026.",
            "Actual latency depends on embedding model size, vector DB sharding, and GPU hardware topology (e.g., NVIDIA L4 vs A100 vs H100).",
            "Fine-Tuning dataset curation quality has a dominant impact on downstream hallucination rates.",
        ]

        return ResearchReport(
            title=f"Technical Evaluation & Comparative Architecture Report: {query}",
            executive_summary=(
                f"This research report evaluates '{query}' through a multi-agent investigation. "
                "The findings distinguish between parametric adaptation (Fine-Tuning) and external context injection (RAG). "
                "For customer support workloads where policies, order statuses, and FAQs change dynamically, RAG provides "
                "superior auditability and near-zero knowledge update latency. Fine-tuning offers value primarily in stylistic "
                "alignment, domain vocabulary compression, and strict formatting guarantees."
            ),
            key_findings=findings_bullets,
            comparative_analysis=(
                "Architecture comparison demonstrates that RAG and Fine-Tuning solve fundamentally orthogonal problems: "
                "RAG provides dynamic information retrieval and hallucination mitigation via grounded references, whereas "
                "Fine-Tuning adjusts the model's behavioral distribution, vocabulary, and token output characteristics. "
                "In customer support scenarios, relying solely on fine-tuning introduces severe risks of hallucinated policy details "
                "and expensive retraining cycles whenever documentation evolves."
            ),
            trade_offs=trade_offs,
            recommendations=recommendations,
            limitations=limitations,
            metadata=metadata or {},
        )


def generate_structured_output(
    schema: Type[T],
    prompt: str,
    settings: Settings,
    system_prompt: Optional[str] = None,
    mock_generator_fn: Optional[Any] = None,
) -> T:
    """Generate a validated Pydantic model using the configured mode.

    Gracefully falls back to mock logic if in Mock Mode or if external model is unreachable.
    """
    if settings.is_mock():
        if mock_generator_fn:
            return mock_generator_fn()
        raise ValueError("Mock generator function must be provided in Mock Mode.")

    chat_model = get_chat_model(settings)
    if not chat_model:
        if mock_generator_fn:
            return mock_generator_fn()
        raise RuntimeError("No chat model available and no mock fallback provided.")

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    # Attempt structured output invocation
    try:
        structured_llm = chat_model.with_structured_output(schema)
        result = structured_llm.invoke(messages)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
    except Exception as exc:
        logger.warning(
            "with_structured_output failed (%s), attempting raw JSON parse fallback: %s",
            type(exc).__name__,
            exc,
        )

    # Fallback: Plain invocation with JSON extraction
    try:
        raw_response = chat_model.invoke(messages)
        text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        # Extract json code block if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            return schema.model_validate(data)

        # Extract bare JSON object
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            data = json.loads(brace_match.group(1))
            return schema.model_validate(data)
    except Exception as exc:
        logger.error("JSON fallback parsing failed: %s", exc)

    # If all remote attempts fail, fall back to mock generator if available
    if mock_generator_fn:
        logger.warning("Remote inference failed or malformed; falling back to deterministic mock response.")
        return mock_generator_fn()

    raise RuntimeError(f"Failed to generate structured output for {schema.__name__}")
