# AgentForge — AI Engineer Technical Interview Guide

This guide is designed to help you confidently present and defend **AgentForge** during an AI Engineer technical interview. Every answer directly references the codebase implementation.

---

## 1. The 60-Second Elevator Pitch

> *"AgentForge is an experimental AI research assistant I built to explore multi-agent systems and modern LLM serving in practice. Rather than using a single monolithic prompt, the system implements a **Supervisor-Worker architecture** in **LangGraph**. A central Supervisor decomposes research inquiries into discrete tasks and delegates them to specialized worker sub-agents—specifically a **Researcher** that queries grounded technical facts and an **Analyst** that evaluates engineering trade-offs.
> 
> These independent sub-agents execute in parallel, and their deliverables converge on a **Critic sub-agent** that acts as an automated quality gate. If the Critic detects omissions, it uses conditional routing to trigger a bounded revision cycle before a **Synthesizer** compiles the final technical report.
> 
> Architecturally, I also wanted to explore decoupled inference: the application integrates models via an OpenAI-compatible API and supports **Mock Mode** for deterministic offline demos, remote hosted endpoints, or high-throughput local inference with **vLLM** using PagedAttention. It's a focused engineering experiment demonstrating orchestration, state management, and inference separation."*

---

## 2. Architecture Walkthrough

| Agent | Module | Core Responsibility | Key Implementation Detail |
| :--- | :--- | :--- | :--- |
| **Supervisor** | `agents/supervisor.py` | Analyzes request, creates execution plan, delegates bounded tasks. | Generates an `ExecutionPlan` with `ResearchTask` objects; routes workflow based on Critic feedback. |
| **Researcher** | `agents/researcher.py` | Factual retrieval, technical specifications, grounded examples. | Queries `KnowledgeRetriever` (`tools/research_tools.py`); explicitly distinguishes retrieved facts from model knowledge. |
| **Analyst** | `agents/analyst.py` | Comparative trade-off evaluation (latency, cost, complexity, freshness). | Evaluates multi-dimensional trade-offs independently of the Researcher. |
| **Critic / Reviewer** | `agents/critic.py` | Automated quality control gatekeeper. | Computes a `completeness_score` (0.0–1.0) and determines `needs_revision`. |
| **Revision Node** | `graph/nodes.py` | Incorporates Critic feedback into deliverables. | Increments `revision_count` and loops back to Critic (capped by `MAX_REVISION_ATTEMPTS`). |
| **Synthesizer** | `agents/synthesizer.py` | Compiles final executive research report. | Merges approved deliverables into structured sections (Executive Summary, Key Findings, Trade-offs, Recommendations). |

---

## 3. Core Technical Concepts Cheat Sheet

- **Supervisor-Worker Pattern**: A hierarchical pattern where a parent supervisor decomposes goals and coordinates worker agents rather than performing work itself.
- **Agent Delegation**: The explicit assignment of bounded tasks with distinct inputs, parameters, and expected schemas to specific agents.
- **Sub-agents**: Worker agents that execute within a constrained scope and return structured Pydantic outputs back to the graph state.
- **Task Decomposition**: Breaking a high-level ambiguous prompt into independent, parallelizable sub-tasks.
- **Parallel Execution**: Concurrently dispatching non-dependent nodes (`supervisor -> researcher` and `supervisor -> analyst`) in LangGraph to minimize wall-clock latency.
- **Fan-in Synchronization**: Graph node (`critic`) waiting until all incoming concurrent branches complete before firing.
- **Conditional Edges**: Graph routing based on runtime state values (e.g. `route_after_critic` inspecting `critic_review.needs_revision`).
- **Centralized State Management**: Shared `AgentForgeState` TypedDict where immutable and appended data (`operator.add` for execution events) are tracked.
- **Structured Outputs**: Using Pydantic models for strict type validation and reliable schema contracts across agent handoffs.
- **LLM Inference**: The raw computation of generating tokens by passing activations through model weights.
- **Model Serving (vLLM)**: High-performance infrastructure software managing GPU memory (PagedAttention), request queuing, continuous batching, and HTTP API exposure.
- **OpenAI-Compatible API**: The standard `/v1/chat/completions` REST protocol allowing interchangeable switching between OpenAI, local vLLM, Ollama, Groq, or Together AI without modifying application code.

---

## 4. 15 Key Interview Questions & Answers

### Q1: Why did you choose a multi-agent architecture instead of a single prompt?
**Answer:**
A single monolithic prompt attempting to research facts, analyze trade-offs, critique its own reasoning, and format an executive summary suffers from **attention dilution**, **conflicting system instructions**, and **hallucination amplification**. 
By decomposing the problem into specialized sub-agents, each agent receives a bounded system prompt and focused schema. The Researcher focuses strictly on factual grounding, the Analyst focuses on architectural trade-offs, and the Critic provides an independent quality check. This separation makes each step testable, debuggable, and allows independent sub-tasks to run in parallel.

---

### Q2: Why not just use a single ReAct agent with tools?
**Answer:**
ReAct (Reason + Act) agents are sequential by nature: they reason, call one tool, observe, and repeat. For research questions like *"Compare RAG vs Fine-Tuning"*, factual discovery and comparative trade-off analysis do not depend on each other sequentially. In a multi-agent graph with LangGraph, we can dispatch the Researcher and Analyst in parallel, cutting wall-clock execution time roughly in half. Furthermore, single ReAct agents frequently suffer from **self-confirmation bias** when evaluating their own output; an independent Critic node provides an objective quality gate.

---

### Q3: What is "delegation" and how is it implemented in your code?
**Answer:**
Delegation is the explicit handoff of a bounded task from an orchestrator to a specialized worker. In AgentForge, delegation is not just a role-play prompt. In `agents/supervisor.py`, the Supervisor parses the query and returns an `ExecutionPlan` containing explicit `ResearchTask` models with unique task IDs, assigned agents, and specific focus parameters. In `graph/workflow.py`, the LangGraph engine uses this plan to route control to the `researcher` and `analyst` nodes, logging explicit `task_delegated` events to the timeline.

---

### Q4: What is the difference between an agent and a sub-agent?
**Answer:**
In modern agentic systems, an **Agent** typically has open-ended authority, planning autonomy, and direct interaction with the user or external systems. A **Sub-agent** is a worker spawned or coordinated by an agent, possessing a bounded scope, specific input/output schemas, and limited autonomy. In AgentForge, the Supervisor acts as the primary orchestrator, while the Researcher, Analyst, Critic, and Synthesizer act as sub-agents executing bounded sub-graphs or functional transformations.

---

### Q5: What does the Supervisor actually do in your implementation?
**Answer:**
The Supervisor performs three key functions:
1. **Decomposition**: Translates the user's inquiry into an `ExecutionPlan` with discrete `ResearchTask` items.
2. **Coordination**: Determines which sub-agents are needed and provides their initial execution context.
3. **Workflow Routing**: After the Critic evaluates the deliverables, the Supervisor evaluates the Critic's review and the current `revision_count` to decide whether to route the state to `revision` or `synthesizer`.

---

### Q6: How does parallel execution work in LangGraph?
**Answer:**
In LangGraph, parallel execution is defined by graph topology. In `graph/workflow.py`, the `supervisor` node has outgoing edges to both `researcher` and `analyst`. When `supervisor` completes, LangGraph dispatches both nodes concurrently. Both workers update independent fields of the centralized state (`research_result` and `analysis_result`), while execution events are merged via LangGraph's `Annotated[List[ExecutionEvent], operator.add]` reducer. LangGraph automatically synchronizes incoming edges at the `critic` node, waiting for both branches to finish before executing the critique.

---

### Q7: How do you prevent infinite loops between the Critic and Revision nodes?
**Answer:**
By enforcing a hard boundary in state:
1. `AgentForgeState` tracks an explicit integer `revision_count`.
2. Central configuration defines `MAX_REVISION_ATTEMPTS` (default 1).
3. The routing function `route_after_critic(state)` checks:
   ```python
   if review.needs_revision and revision_count < max_revisions:
       return "revision"
   return "synthesizer"
   ```
Even if the Critic remains unsatisfied, once `revision_count >= max_revisions`, the workflow forces a transition to the `synthesizer` node, preventing infinite execution and uncontrolled API costs.

---

### Q8: What happens if an agent fails or produces malformed JSON?
**Answer:**
AgentForge implements multiple layers of defensive programming:
1. **Schema Validation**: All agent responses are parsed into Pydantic models.
2. **Regex JSON Fallback**: If `with_structured_output` fails or the model returns markdown JSON blocks (````json ... ````), regex extraction parses the payload.
3. **Deterministic Mock Fallback**: In `core/llm.py`, if a remote model drops the connection or produces invalid output, `generate_structured_output` catches the exception and falls back to a deterministic domain generator.
4. **State Isolation**: Sub-agent exceptions are caught and logged as `EventType.ERROR` in `AgentForgeState["errors"]` rather than crashing the orchestrator process.

---

### Q9: Why did you use LangGraph instead of vanilla Python or standard LangChain chains?
**Answer:**
Standard LangChain chains (`RunnableSequence`) represent Directed Acyclic Graphs (DAGs) with linear flow. They cannot naturally represent **cyclic loops** (like the Critic -> Revision -> Critic feedback loop) or complex multi-agent state handoffs. LangGraph treats workflows as state machines with first-class support for:
- Cyclic graphs and conditional branch routing.
- Synchronized parallel branch fan-out and fan-in.
- Centralized checkpointing and state persistence.
- Clear separation between state definitions and node execution logic.

---

### Q10: What is the difference between LangChain and LangGraph?
**Answer:**
- **LangChain**: A library of component abstractions (prompt templates, chat models, document loaders, output parsers, tools) designed to integrate LLMs into Python applications.
- **LangGraph**: An orchestration framework built on top of LangChain that models complex multi-agent workflows as stateful, directed graphs supporting parallel execution, cycles, human-in-the-loop checkpoints, and multi-actor state synchronization.
In AgentForge: LangChain wraps the LLMs and schemas; LangGraph coordinates the workflow.

---

### Q11: What is vLLM, and how does it fit into this project?
**Answer:**
**vLLM** is an open-source, high-throughput LLM serving engine. It is famous for introducing **PagedAttention**, an algorithm that treats the Attention Key-Value (KV) cache like virtual memory pages in an operating system, virtually eliminating GPU VRAM fragmentation.
In AgentForge, vLLM acts as an optional local model serving engine. When a user runs open models locally (e.g. Llama-3.1-8B-Instruct), vLLM handles continuous batching and serves the model over an OpenAI-compatible REST endpoint (`http://localhost:8000/v1`). LangGraph then orchestrates the agents against this high-throughput backend.

---

### Q12: What is the architectural difference between Inference and Orchestration?
**Answer:**
This is a critical distinction:
- **Orchestration (LangGraph)**: The business logic that determines *who* runs, *when* they run, *what state* is passed, and *how decisions are routed*. It operates on task graphs and state dictionaries.
- **Inference (vLLM / LLM)**: The low-level mathematical computation that converts a prompt into tokens by computing forward passes through transformer layers on hardware (GPUs/TPUs).
Confusing the two is a common junior mistake. vLLM is not an agent framework, and LangGraph does not compute tensor matrices.

---

### Q13: Why use an OpenAI-compatible API instead of vendor-specific SDKs?
**Answer:**
Using the OpenAI-compatible standard (`/v1/chat/completions`) completely decouples the agent orchestration logic from the underlying model hosting provider. In AgentForge, the exact same agent code runs against:
1. `MockMode` (local synthetic testing)
2. `http://localhost:8000/v1` (locally hosted vLLM with GPU)
3. OpenAI, Together AI, Groq, or OpenRouter
Switching backends requires changing only two environment variables (`LLM_BASE_URL` and `LLM_MODEL`) without rewriting agent prompts or graph nodes.

---

### Q14: What are the main disadvantages and costs of multi-agent systems?
**Answer:**
1. **Token Cost Multipliers**: Decomposing a task into 5 agents results in multiple LLM invocations per user question, significantly increasing token usage.
2. **Cumulative Latency**: Sequential steps (Supervisor -> Critic -> Synthesizer) compound API latency.
3. **State Complexity**: Debugging state transitions and edge routing is significantly harder than debugging a single script.
4. **Nondeterministic Compounding**: Errors or subtle hallucinations in early worker agents (Researcher) can mislead downstream agents (Analyst and Synthesizer).

---

### Q15: When would you recommend AGAINST using multiple agents?
**Answer:**
I would avoid a multi-agent architecture when:
1. **The task is narrow, linear, or deterministic**: e.g., summarizing a single document, translating text, or extracting regex-friendly JSON. A single prompt with few-shot examples or structured outputs is faster, cheaper, and more reliable.
2. **Strict sub-second latency is required**: Agent graphs involve multiple network hops and generation cycles that typically take several seconds.
3. **Budget constraints dominate**: Monolithic prompts or fine-tuned smaller models yield far higher cost efficiency when token budgets are constrained.
Multi-agent systems should be reserved for high-uncertainty tasks requiring task decomposition, concurrent multi-source retrieval, and rigorous multi-stage validation.
