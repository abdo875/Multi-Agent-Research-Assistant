"""Agent status indicators component for AgentForge UI.

Renders minimal, professional status cards for the five specialized agents.
"""

from typing import Dict, List, Optional
import streamlit as st
from schemas.events import ExecutionEvent


def get_agent_status_map(events: List[ExecutionEvent]) -> Dict[str, Dict[str, str]]:
    """Determine live status and latest action for each agent from execution events."""
    agents = {
        "Supervisor": {"status": "Idle", "icon": "👑", "color": "#64748b", "action": "Awaiting request"},
        "Researcher": {"status": "Idle", "icon": "🔍", "color": "#64748b", "action": "Awaiting task"},
        "Analyst": {"status": "Idle", "icon": "📊", "color": "#64748b", "action": "Awaiting task"},
        "Critic": {"status": "Idle", "icon": "⚖️", "color": "#64748b", "action": "Awaiting deliverables"},
        "Synthesizer": {"status": "Idle", "icon": "📝", "color": "#64748b", "action": "Awaiting approval"},
    }

    if not events:
        return agents

    for e in events:
        name = e.agent_name
        if name in agents:
            if e.event_type.value in ["task_completed", "plan_created", "report_generated", "routing"]:
                agents[name]["status"] = "Completed"
                agents[name]["color"] = "#10b981"  # Emerald green
                agents[name]["action"] = e.summary
            elif e.event_type.value in ["task_started", "task_delegated"]:
                agents[name]["status"] = "Active"
                agents[name]["color"] = "#3b82f6"  # Royal blue
                agents[name]["action"] = e.summary
            elif e.event_type.value == "evaluation":
                agents[name]["status"] = "Reviewed"
                agents[name]["color"] = "#10b981" if "Approved" in e.summary else "#f59e0b"
                agents[name]["action"] = e.summary
            elif e.event_type.value == "revision_requested":
                agents[name]["status"] = "Revising"
                agents[name]["color"] = "#f59e0b"  # Amber
                agents[name]["action"] = e.summary

    return agents


def render_agent_status_grid(events: Optional[List[ExecutionEvent]] = None) -> None:
    """Render a 5-column grid of agent status cards."""
    status_map = get_agent_status_map(events or [])

    cols = st.columns(5)
    for col, (agent_name, data) in zip(cols, status_map.items()):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: #1e293b;
                    border: 1px solid #334155;
                    border-top: 3px solid {data['color']};
                    border-radius: 8px;
                    padding: 12px 14px;
                    min-height: 125px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 1.1rem;">{data['icon']} <b>{agent_name}</b></span>
                        <span style="
                            font-size: 0.72rem;
                            padding: 2px 8px;
                            border-radius: 12px;
                            background: {data['color']}22;
                            color: {data['color']};
                            font-weight: 600;
                            border: 1px solid {data['color']}44;
                        ">{data['status']}</span>
                    </div>
                    <p style="
                        font-size: 0.78rem;
                        color: #94a3b8;
                        margin: 0;
                        line-height: 1.35;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        display: -webkit-box;
                        -webkit-line-clamp: 3;
                        -webkit-box-orient: vertical;
                    ">{data['action']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
