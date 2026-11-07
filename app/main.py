"""AgentForge - Experimental Multi-Agent Research Assistant.

Streamlit application demonstrating Supervisor-Worker delegation, sub-agents,
LangGraph orchestration, and vLLM inference connectivity.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from app.components.agent_status import render_agent_status_grid
from app.components.report_view import render_critic_review, render_final_report
from app.components.timeline import render_execution_plan, render_timeline
from core.config import AppMode, Settings, get_settings
from core.inference import check_inference_health, get_inference_summary
from graph.workflow import create_agentforge_graph

# Page configuration
st.set_page_config(
    page_title="AgentForge | Multi-Agent Research Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Navy/Blue Theme Styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
        }
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border: 1px solid #3b82f6;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1.25rem;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
            border-color: #60a5fa;
            box-shadow: 0 0 12px rgba(37, 99, 235, 0.4);
        }
        .stTextArea textarea {
            background-color: #1e293b;
            color: #f8fafc;
            border: 1px solid #334155;
            border-radius: 6px;
        }
        .stTextArea textarea:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 1px #3b82f6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_active_settings() -> Settings:
    """Read session state or environment settings."""
    env_settings = get_settings()

    mode_choice = st.session_state.get("selected_mode", env_settings.app_mode.value)
    base_url = st.session_state.get("custom_base_url", env_settings.llm_base_url)
    model_name = st.session_state.get("custom_model", env_settings.llm_model)
    api_key = st.session_state.get("custom_api_key", env_settings.llm_api_key)

    return Settings(
        app_mode=AppMode(mode_choice),
        llm_base_url=base_url,
        llm_model=model_name,
        llm_api_key=api_key or None,
        max_revision_attempts=1,
    )


def main():
    # Header Branding
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #60a5fa;">AgentForge</h1>
                <span style="
                    background: #1e3a8a;
                    color: #93c5fd;
                    padding: 4px 10px;
                    border-radius: 12px;
                    font-size: 0.75rem;
                    font-weight: 600;
                ">EXPERIMENTAL AI ASSISTANT</span>
            </div>
            <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 0.95rem;">
                Experimenting with Multi-Agent AI Architectures, Supervisor Delegation, LangGraph, and Local vLLM Inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Inference & Execution Controls
    with st.sidebar:
        st.markdown("### ⚙️ Inference Settings")

        mode_options = [AppMode.MOCK.value, AppMode.VLLM.value, AppMode.HOSTED.value]
        selected_mode = st.selectbox(
            "Execution Mode",
            mode_options,
            index=0,
            key="selected_mode",
            help="Mock: Zero API key needed, fast deterministic demo. vLLM: Local model server at port 8000. Hosted: Remote OpenAI-compatible API.",
        )

        settings = get_active_settings()
        inf_summary = get_inference_summary(settings)

        # Dynamic fields based on mode
        if selected_mode == AppMode.VLLM.value:
            st.text_input("vLLM Base URL", value=settings.llm_base_url, key="custom_base_url")
            st.text_input("Model Name", value=settings.llm_model, key="custom_model")
            st.caption("Default local endpoint: `http://localhost:8000/v1`")
        elif selected_mode == AppMode.HOSTED.value:
            st.text_input("API Base URL", value="https://api.openai.com/v1", key="custom_base_url")
            st.text_input("Model Name", value="gpt-4o-mini", key="custom_model")
            st.text_input("API Key", type="password", key="custom_api_key")

        # Inference diagnostic card
        st.markdown("---")
        st.markdown("##### 🔌 Inference Diagnostics")
        status_color = "#10b981" if "Ready" in inf_summary["status"] or "Online" in inf_summary["status"] else "#ef4444"
        st.markdown(
            f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px 12px; font-size: 0.8rem;">
                <div><b>Provider:</b> {inf_summary['provider']}</div>
                <div><b>Status:</b> <span style="color: {status_color}; font-weight: 600;">{inf_summary['status']}</span></div>
                <div><b>Endpoint:</b> <code style="font-size: 0.75rem;">{inf_summary['endpoint']}</code></div>
                <div style="margin-top: 6px; color: #94a3b8; font-size: 0.75rem;">{inf_summary['details']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Interview Demo Controls
        st.markdown("---")
        st.markdown("##### 🧪 Demo Controls")
        force_revision = st.checkbox(
            "Simulate Critic Revision Loop",
            value=False,
            help="Triggers a simulated quality gap to demonstrate LangGraph conditional routing and self-correction during the interview.",
        )

        if st.button("Reset State", use_container_width=True):
            st.session_state.pop("workflow_output", None)
            st.rerun()

    # Main Research Input Area
    st.markdown("##### 🎯 Research Inquiry")

    default_query = "Compare RAG and Fine-Tuning for a customer support application."
    query = st.text_area(
        "Enter your technical research topic or question:",
        value=st.session_state.get("research_query", default_query),
        height=75,
        key="research_query",
    )

    # Example query quick-selectors
    st.markdown(
        "<span style='font-size: 0.8rem; color: #94a3b8;'>Quick Demonstration Topics:</span>",
        unsafe_allow_html=True,
    )
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("📋 RAG vs Fine-Tuning", use_container_width=True):
            st.session_state["research_query"] = "Compare RAG and Fine-Tuning for a customer support application."
            st.rerun()
    with col_ex2:
        if st.button("🚀 vLLM vs Transformers Serving", use_container_width=True):
            st.session_state["research_query"] = "Evaluate vLLM PagedAttention versus standard Hugging Face Transformers for multi-agent serving."
            st.rerun()
    with col_ex3:
        if st.button("🤖 Multi-Agent Orchestration", use_container_width=True):
            st.session_state["research_query"] = "Analyze Supervisor-Worker multi-agent architectures versus monolithic LLM chains."
            st.rerun()

    run_btn = st.button("⚡ Run Multi-Agent Research", use_container_width=True)

    # Workflow Execution
    if run_btn:
        with st.spinner("Orchestrating multi-agent workflow (Supervisor -> Workers -> Critic -> Synthesizer)..."):
            graph = create_agentforge_graph()

            initial_state = {
                "user_query": query,
                "execution_plan": None,
                "assigned_tasks": [],
                "research_result": None,
                "analysis_result": None,
                "critic_review": None,
                "revision_count": 0,
                "final_report": None,
                "execution_events": [],
                "errors": [],
                "force_revision_test": force_revision,
            }

            try:
                output = graph.invoke(initial_state)
                st.session_state["workflow_output"] = output
                st.toast("Multi-agent research workflow completed!", icon="✅")
            except Exception as exc:
                st.error(f"Workflow execution failed: {type(exc).__name__}: {exc}")

    # Output Presentation
    output = st.session_state.get("workflow_output")
    events = output.get("execution_events", []) if output else []

    st.markdown("---")
    st.markdown("### 🤖 Multi-Agent Live Activity")
    render_agent_status_grid(events)

    if output:
        # Display Execution Plan
        st.markdown("---")
        render_execution_plan(output.get("execution_plan"))

        # Two-column layout for Timeline and Critic Review
        st.markdown("---")
        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            render_timeline(events)

        with col_right:
            st.markdown("##### ⚖️ Critic Review & Quality Gate")
            render_critic_review(output.get("critic_review"))

        # Final Report
        render_final_report(output.get("final_report"))


if __name__ == "__main__":
    main()
