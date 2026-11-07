"""Execution timeline and execution plan visualization components."""

from typing import List, Optional
import streamlit as st
from schemas.events import ExecutionEvent
from schemas.task import ExecutionPlan


def render_execution_plan(plan: Optional[ExecutionPlan]) -> None:
    """Render the Supervisor's decomposed execution plan."""
    if not plan:
        return

    st.markdown(
        f"""
        <div style="
            background: #0f172a;
            border: 1px solid #1e3a8a;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 20px;
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-size: 1.05rem; font-weight: 600; color: #93c5fd;">
                    👑 Supervisor Execution Plan
                </span>
                <span style="
                    font-size: 0.75rem;
                    background: #1e3a8a;
                    color: #bfdbfe;
                    padding: 3px 10px;
                    border-radius: 12px;
                    font-weight: 500;
                ">
                    Strategy: {plan.execution_strategy.replace('_', ' ').title()}
                </span>
            </div>
            <p style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 8px;">
                <b>Objective:</b> {plan.objective}
            </p>
            <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 12px;">
                <b>Rationale:</b> {plan.rationale}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Delegated Worker Tasks")
    cols = st.columns(len(plan.tasks))
    for col, task in zip(cols, plan.tasks):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    padding: 12px;
                    height: 100%;
                ">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-size: 0.75rem; font-family: monospace; color: #60a5fa;">{task.task_id}</span>
                        <span style="font-size: 0.72rem; color: #a7f3d0; background: #064e3b; padding: 1px 6px; border-radius: 4px;">
                            {task.assigned_agent}
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; color: #f1f5f9; font-weight: 500; margin-bottom: 4px;">
                        {task.task_type.value.title()} Task
                    </div>
                    <p style="font-size: 0.78rem; color: #94a3b8; margin: 0; line-height: 1.35;">
                        {task.description}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_timeline(events: List[ExecutionEvent]) -> None:
    """Render the chronological execution event timeline."""
    if not events:
        st.info("No workflow events logged yet.")
        return

    st.markdown("##### ⏱️ Delegation & Execution Timeline")

    event_badge_styles = {
        "plan_created": ("#1e3a8a", "#bfdbfe", "Plan Created"),
        "task_delegated": ("#1e3a8a", "#bfdbfe", "Delegated"),
        "task_started": ("#0369a1", "#bae6fd", "Started"),
        "task_completed": ("#065f46", "#a7f3d0", "Completed"),
        "evaluation": ("#78350f", "#fde68a", "Evaluated"),
        "routing": ("#4c1d95", "#ddd6fe", "Routing"),
        "revision_requested": ("#831843", "#fbcfe8", "Revision Mandate"),
        "report_generated": ("#14532d", "#bbf7d0", "Synthesized"),
        "error": ("#7f1d1d", "#fecaca", "Error"),
    }

    for ev in events:
        bg, fg, label = event_badge_styles.get(
            ev.event_type.value, ("#334155", "#e2e8f0", ev.event_type.value.title())
        )

        with st.container():
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: flex-start;
                    gap: 12px;
                    padding: 8px 12px;
                    margin-bottom: 6px;
                    background: #111827;
                    border: 1px solid #1f2937;
                    border-radius: 6px;
                ">
                    <span style="
                        font-family: monospace;
                        font-size: 0.75rem;
                        color: #64748b;
                        white-space: nowrap;
                        padding-top: 2px;
                    ">{ev.timestamp}</span>

                    <span style="
                        font-size: 0.72rem;
                        font-weight: 600;
                        background: {bg};
                        color: {fg};
                        padding: 2px 8px;
                        border-radius: 4px;
                        white-space: nowrap;
                    ">{ev.agent_name}</span>

                    <span style="
                        font-size: 0.72rem;
                        color: #94a3b8;
                        background: #1e293b;
                        padding: 2px 6px;
                        border-radius: 4px;
                        white-space: nowrap;
                    ">{label}</span>

                    <div style="flex-grow: 1; font-size: 0.82rem; color: #e2e8f0; line-height: 1.4;">
                        <b>{ev.summary}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
