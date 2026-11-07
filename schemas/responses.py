"""Response schemas for specialized sub-agents and final synthesis.

Enforces structured outputs across Researcher, Analyst, Critic, and Synthesizer.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ResearchFinding(BaseModel):
    """An individual technical concept or fact retrieved by the Researcher."""

    topic: str = Field(description="Name or topic of the technical concept")
    facts: List[str] = Field(description="Key technical facts and specifications")
    technical_details: str = Field(description="Deeper explanation of architecture, mechanism, or operation")
    examples: List[str] = Field(default_factory=list, description="Concrete examples or benchmark references")
    source_type: str = Field(
        default="retrieved_reference",
        description="Origin of knowledge: 'retrieved_reference' or 'parametric_knowledge'",
    )


class ResearchResult(BaseModel):
    """Structured output returned by the Researcher sub-agent."""

    task_id: str = Field(description="ID of the research task fulfilled")
    findings: List[ResearchFinding] = Field(description="List of researched concepts")
    sources_consulted: List[str] = Field(
        default_factory=list,
        description="Tools, knowledge domains, or references inspected",
    )
    summary: str = Field(description="Concise synthesis of research findings")


class AnalysisDimension(BaseModel):
    """Comparative analysis along a specific technical dimension."""

    dimension: str = Field(description="Evaluated dimension (e.g. Latency, Cost, Accuracy, Engineering Complexity)")
    comparison: str = Field(description="Direct comparison between evaluated options")
    trade_offs: str = Field(description="Inherent technical trade-offs for this dimension")


class AnalysisResult(BaseModel):
    """Structured output returned by the Analyst sub-agent."""

    task_id: str = Field(description="ID of the analysis task fulfilled")
    insights: List[str] = Field(description="Key strategic and technical insights")
    comparisons: List[AnalysisDimension] = Field(description="Detailed multi-dimensional comparisons")
    trade_offs_summary: str = Field(description="High-level trade-off summary")
    practical_applications: List[str] = Field(
        description="Recommended real-world use cases and industry scenarios"
    )


class CriticReview(BaseModel):
    """Structured evaluation returned by the Critic / Reviewer sub-agent."""

    objective_satisfied: bool = Field(description="Whether the user's core objective was directly satisfied")
    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Completeness and rigor score from 0.0 (incomplete) to 1.0 (thorough)",
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Key gaps, missing dimensions, or omitted comparisons",
    )
    contradictions_detected: List[str] = Field(
        default_factory=list,
        description="Inconsistencies or conflicting statements between research and analysis",
    )
    quality_assessment: str = Field(description="Overall qualitative critique")
    needs_revision: bool = Field(description="Whether a revision loop is mandated before synthesis")
    revision_instructions: Optional[str] = Field(
        default=None,
        description="Specific instructions for agents if revision is required",
    )


class ResearchReport(BaseModel):
    """Final comprehensive research report produced by the Synthesizer sub-agent."""

    title: str = Field(description="Title of the research report")
    executive_summary: str = Field(description="High-level executive summary")
    key_findings: List[str] = Field(description="Bulleted core findings from research")
    comparative_analysis: str = Field(description="Synthesized comparative evaluation")
    trade_offs: List[str] = Field(description="Critical engineering trade-offs to consider")
    recommendations: List[str] = Field(description="Actionable guidance and decision framework")
    limitations: List[str] = Field(description="Assumptions, scope limitations, and knowledge boundaries")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (agents involved, revision cycles, inference mode)",
    )
