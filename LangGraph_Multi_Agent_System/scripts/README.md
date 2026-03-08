# LangGraph Multi-Agent System — Pattern Scripts

## Architecture Rule

```
root module/          ← COMPONENT IMPLEMENTATION (reusable library)
scripts/<area>/       ← PATTERN DEMONSTRATIONS (how to use the library)
```

Every script in `scripts/` is a **pattern demo**. It imports from the root
component modules (`guardrails/`, `hitl/`, `memory/`, `orchestration/`,
`resilience/`, `core/`, `tools/`, `observability/`, `agents/`) but does NOT
reimplement their logic. The root modules are the authoritative source.

---

## Master Learning Path — 9 Areas in Order

Study the areas in this sequence. Each builds on the previous.

```
Area 1: Tools & Tool Binding         scripts/tools/
   ↓
Area 2: Handoff Patterns             scripts/handoff/
   ↓
Area 3: Guardrails                   scripts/guardrails/
   ↓
Area 4: Human-in-the-Loop (HITL)     scripts/HITL/
   ↓
Area 5: Memory Management            scripts/memory_management/
   ↓
Area 6: Orchestration Patterns       scripts/orchestration/
   ↓
Area 7: MAS Architectures            scripts/MAS_architectures/
   ↓
Area 8: Observability & Traceability scripts/observability_and_traceability/
   ↓
Area 9: Resilience (embedded)        resilience/ + orchestration/
```

> **Resilience** has no standalone pattern scripts by design. It is embedded
> inside the orchestration component module. See `resilience/__init__.py` for
> the explanation and `orchestration/orchestrator.py` for integration.

---

## Area 1 — Tools & Tool Binding

**Root module:** `tools/`  
**When you've finished:** You understand `bind_tools()`, `ToolNode`, structured
output from LLMs, dynamic tool selection, and graceful error handling.

| Script | Use when |
|--------|----------|
| `tool_binding.py` | Starting point — learn how to scope tools per-agent |
| `toolnode_patterns.py` | Choosing between ToolNode-as-node vs ToolNode.invoke() |
| `structured_output.py` | You need validated JSON from the LLM, not free text |
| `dynamic_tool_selection.py` | Tool set depends on runtime input (not fixed at build time) |
| `tool_error_handling.py` | Tools can fail — learn how to recover gracefully |

---

## Area 2 — Handoff Patterns

**Root modules:** `core/`, `tools/`, `observability/`  
**When you've finished:** You understand fixed-edge pipelines, conditional
routing, LLM-driven Command handoffs, supervisor-coordinated workers, depth
guards, and parallel fan-out/merge.

| Script | Use when |
|--------|----------|
| `linear_pipeline.py` | Execution order is always fixed (baseline pattern) |
| `conditional_routing.py` | Routing depends on output (Python function decides, zero LLM cost) |
| `command_handoff.py` | The LLM itself should decide which agent runs next |
| `supervisor.py` | One coordinator LLM routes multiple workers dynamically |
| `multihop_depth_guard.py` | Command-based chains risk infinite loops — add depth limits |
| `parallel_fanout.py` | Multiple agents should run concurrently, results merged |

---

## Area 3 — Guardrails

**Root module:** `guardrails/`  
**When you've finished:** You understand binary and three-way routing with
validation, confidence gating, LLM-as-judge, and layered validation stacks.

| Script | Use when |
|--------|----------|
| `input_validation.py` | Binary pass/fail routing before the agent runs |
| `output_validation.py` | Validate agent output before it leaves the pipeline |
| `confidence_gating.py` | LLM reports its own confidence; gate low-confidence responses |
| `llm_as_judge.py` | A second LLM evaluates the first LLM's output semantically |
| `layered_validation.py` | Stack input + output guardrails for defence-in-depth |

---

## Area 4 — Human-in-the-Loop (HITL)

**Root module:** `hitl/`  
**When you've finished:** You understand `interrupt()` / `Command(resume=)`,
approval flows, edit-before-approve, multi-step approvals, and escalation chains.

| Script | Use when |
|--------|----------|
| `basic_approval.py` | Start here — core interrupt/resume mechanics |
| `tool_call_confirmation.py` | A human must confirm before a tool is executed |
| `edit_before_approve.py` | Human can modify the agent's output before approving |
| `multi_step_approval.py` | Multiple sequential checkpoints within one workflow |
| `escalation_chain.py` | Junior review first, escalate to senior if rejected |

---

## Area 5 — Memory Management

**Root module:** `memory/`  
**When you've finished:** You understand working memory (scratchpad), semantic
retrieval (RAG), checkpoint persistence, conversation summarisation, and
shared multi-agent memory.

| Script | Use when |
|--------|----------|
| `working_memory_scratchpad.py` | Agents share a key-value scratchpad through state |
| `semantic_retrieval.py` | Agents query a vector store (ChromaDB RAG) |
| `checkpoint_persistence.py` | State must survive restarts (memory/sqlite/postgres/redis) |
| `conversation_memory.py` | Multi-turn chat with rolling summarisation |
| `shared_memory_multi_agent.py` | All memory tiers combined in one multi-agent workflow |

---

## Area 6 — Orchestration Patterns

**Root modules:** `orchestration/`, `resilience/`  
**When you've finished:** You understand the 5 orchestration patterns and how
resilience (circuit breaker, retry, timeout, rate limiter) is embedded inside
the orchestration layer.

Each pattern follows a 4-file layout: `models.py` → `agents.py` → `graph.py` → `runner.py`

| Pattern (STAGE) | Use when |
|----------------|----------|
| `supervisor_orchestration/` (STAGE 1.x) | Central coordinator routes dynamically |
| `peer_to_peer_orchestration/` (STAGE 2.x) | Agents share findings without a central controller |
| `dynamic_router_orchestration/` (STAGE 3.x) | One-shot LLM classification selects which specialist runs |
| `graph_of_subgraphs_orchestration/` (STAGE 4.x) | Each specialty runs a multi-step subgraph atomically |
| `hybrid_orchestration/` (STAGE 5.x) | Supervisor routes at department level; P2P within departments |

---

## Area 7 — MAS Architectures

**Root modules:** `agents/`, `core/`, `observability/`  
**When you've finished:** You understand 7 higher-level multi-agent system
patterns that compose on top of the orchestration primitives.

| Script (STAGE) | Use when |
|----------------|----------|
| `supervisor_orchestration.py` (STAGE 1.x) | Central coordinator, dynamic routing (foundation pattern) |
| `sequential_pipeline.py` (STAGE 2.x) | Fixed sequential flow with no supervisor |
| `parallel_voting.py` (STAGE 3.x) | Agents vote; majority wins |
| `adversarial_debate.py` (STAGE 4.x) | Two agents argue opposing views; judge rules |
| `hierarchical_delegation.py` (STAGE 5.x) | Multi-level org structure (L3 specialists → L2 leads → L1 exec) |
| `map_reduce_fanout.py` (STAGE 6.x) | Parallel sub-tasks (map) aggregated (reduce) |
| `reflection_self_critique.py` (STAGE 7.x) | Agent critiques its own output and iterates |

---

## Area 8 — Observability & Traceability

**Root module:** `observability/`  
**When you've finished:** You understand trace hierarchies, per-agent cost
tracking, rule-based and LLM-based evaluation, session tracing, and a fully
instrumented production-style pipeline.

| Script (STAGE) | Use when |
|----------------|----------|
| `trace_hierarchy.py` (STAGE 1.x) | Start here — parent/child trace structure |
| `agent_metrics_and_cost.py` (STAGE 2.x) | You need token usage and cost per agent |
| `trace_scoring_and_evaluation.py` (STAGE 3.x) | Automated quality scoring of agent outputs |
| `session_based_tracing.py` (STAGE 4.x) | Group traces by user session for multi-turn analysis |
| `observed_clinical_pipeline.py` (STAGE 5.x) | Full pipeline with all observability patterns combined |

---

## STAGE Numbering Convention

Within each pattern script, sections are labelled:

```
# STAGE X.Y — Section Name
```

- **X** = pattern number within the area (e.g. STAGE 2 = Pattern 2 of the area)
- **Y** = sub-section within the pattern:
  - `.1` — State / Schema Definition
  - `.2` — Setup (tools, agents, registries)
  - `.3` — Node Definitions
  - `.4` — Graph Construction
  - `.5` — Test Cases / Main Execution
  - (additional sub-sections for complex patterns)

In runtime output, `[Step X.Y]` labels are used (not `[STAGE X.Y]`) to avoid
confusion between documentation structure and execution trace.

---

## Running Any Script

```bash
cd "D:/Agentic AI/LangGraph_Multi_Agent_System"
python -m scripts.<area>.<script_name>

# Examples:
python -m scripts.tools.tool_binding
python -m scripts.handoff.supervisor
python -m scripts.guardrails.input_validation
python -m scripts.HITL.basic_approval
python -m scripts.memory_management.conversation_memory
python -m scripts.orchestration.supervisor_orchestration.runner
python -m scripts.MAS_architectures.hierarchical_delegation
python -m scripts.observability_and_traceability.agent_metrics_and_cost
```
