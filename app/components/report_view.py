"""Report and Critic review display components for AgentForge UI."""

from typing import Optional
import streamlit as st
from schemas.responses import CriticReview, ResearchReport


def render_critic_review(review: Optional[CriticReview]) -> None:
    """Render the Critic sub-agent's evaluation card."""
    if not review:
        return

    is_approved = not review.needs_revision
    accent_color = "#10b981" if is_approved else "#f59e0b"
    status_label = "Approved for Synthesis" if is_approved else "Revision Required"

    st.markdown(
        f"""
        <div style="
            background: #0f172a;
            border: 1px solid #1e293b;
            border-left: 4px solid {accent_color};
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 20px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.05rem; font-weight: 600; color: #f8fafc;">
                    ⚖️ Critic Quality Evaluation
                </span>
                <span style="
                    background: {accent_color}22;
                    color: {accent_color};
                    border: 1px solid {accent_color}44;
                    padding: 3px 10px;
                    border-radius: 12px;
                    font-size: 0.78rem;
                    font-weight: 600;
                ">{status_label}</span>
            </div>
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px;">
                    <span>Completeness Score</span>
                    <span style="color: {accent_color}; font-weight: 600;">{review.completeness_score:.0%}</span>
                </div>
                <div style="background: #1e293b; border-radius: 4px; height: 6px; width: 100%;">
                    <div style="background: {accent_color}; height: 6px; border-radius: 4px; width: {review.completeness_score * 100}%;"></div>
                </div>
            </div>
            <p style="font-size: 0.85rem; color: #cbd5e1; margin: 0 0 6px 0;">
                <b>Critique:</b> {review.quality_assessment}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if review.needs_revision and review.revision_instructions:
        st.warning(f"**Revision Mandate:** {review.revision_instructions}")

    if review.missing_information:
        with st.expander("Identified Information Gaps", expanded=False):
            for gap in review.missing_information:
                st.markdown(f"- {gap}")


def render_final_report(report: Optional[ResearchReport]) -> None:
    """Render the final comprehensive research report."""
    if not report:
        return

    st.markdown("---")
    st.markdown(f"### 📄 {report.title}")

    # Metadata badges
    if report.metadata:
        mode_val = report.metadata.get("app_mode", "mock").upper()
        model_val = report.metadata.get("model", "Deterministic Engine")
        rev_val = report.metadata.get("revision_count", 0)
        st.markdown(
            f"""
            <div style="display: flex; gap: 8px; margin-bottom: 16px;">
                <span style="font-size: 0.72rem; background: #1e293b; color: #93c5fd; padding: 2px 8px; border-radius: 4px; border: 1px solid #334155;">
                    MODE: <b>{mode_val}</b>
                </span>
                <span style="font-size: 0.72rem; background: #1e293b; color: #cbd5e1; padding: 2px 8px; border-radius: 4px; border: 1px solid #334155;">
                    MODEL: <b>{model_val}</b>
                </span>
                <span style="font-size: 0.72rem; background: #1e293b; color: #a7f3d0; padding: 2px 8px; border-radius: 4px; border: 1px solid #334155;">
                    REVISION CYCLES: <b>{rev_val}</b>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Executive Summary Card
    st.markdown(
        f"""
        <div style="
            background: #111827;
            border-left: 4px solid #3b82f6;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 20px;
        ">
            <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #60a5fa; margin-bottom: 4px;">
                Executive Summary
            </div>
            <div style="font-size: 0.9rem; color: #e2e8f0; line-height: 1.5;">
                {report.executive_summary}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🔍 Key Factual Findings")
        for finding in report.key_findings:
            st.markdown(f"- {finding}")

    with col2:
        st.markdown("##### ⚖️ Engineering Trade-offs")
        for trade_off in report.trade_offs:
            st.markdown(f"- {trade_off}")

    st.markdown("##### 📊 Comparative Analysis")
    st.write(report.comparative_analysis)

    st.markdown("##### 🎯 Strategic Recommendations")
    for rec in report.recommendations:
        st.markdown(f"1. {rec}")

    if report.limitations:
        with st.expander("Scope Boundaries & Technical Limitations", expanded=False):
            for lim in report.limitations:
                st.markdown(f"- {lim}")
