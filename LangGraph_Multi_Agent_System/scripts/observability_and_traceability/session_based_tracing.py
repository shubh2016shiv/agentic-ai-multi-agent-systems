#!/usr/bin/env python3
"""
============================================================
Session-Based Tracing
============================================================
Pattern 4: Multi-turn session grouping and cross-request
correlation using Langfuse session_id and user_id.
Prerequisite: trace_scoring_and_evaluation.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Clinical interactions are not single requests. A doctor may
have multiple turns with the system:

    Turn 1: "Assess this patient's potassium level"
    Turn 2: "What about the drug interactions?"
    Turn 3: "Generate a discharge summary"

Each turn creates a separate TRACE in Langfuse. But they
all belong to the same SESSION. Langfuse links them via
session_id so you can:

    1. Replay the full session in the dashboard
    2. Aggregate metrics across turns (total cost per session)
    3. Track per-doctor usage (user_id filtering)
    4. Correlate follow-up questions with initial assessments

------------------------------------------------------------
OBSERVABILITY vs TRACEABILITY -- what this covers
------------------------------------------------------------
    TRACEABILITY:
        - Which turns happened in which order
        - What each turn received as input and produced
        - Cross-turn state continuity via checkpoints

    OBSERVABILITY:
        - Per-session cost, latency, token usage
        - Per-doctor usage patterns
        - Session quality (are follow-up turns improving outcomes?)

------------------------------------------------------------
GRAPH TOPOLOGY (same graph, invoked 3 times)
------------------------------------------------------------

    [START]
       |
       v
    [clinical_agent]   <-- LLM call with session_id in config
       |
       v
    [END]

    Turn 1: session_id="session-001", input="initial assessment"
    Turn 2: session_id="session-001", input="drug interactions"
    Turn 3: session_id="session-001", input="discharge summary"

    All three traces grouped under session-001 in Langfuse.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()            graph           Langfuse              MemorySaver
      |                 |                |                       |
      |-- Turn 1 ------>|                |                       |
      |                 |-- LLM call --->|-- Trace (session-001) |
      |                 |                |                       |-- save state
      |<-- result 1 ---|                |                       |
      |                 |                |                       |
      |-- Turn 2 ------>|                |                       |
      |                 |-- LLM call --->|-- Trace (session-001) |
      |                 |                |   (linked to Turn 1)  |-- save state
      |<-- result 2 ---|                |                       |
      |                 |                |                       |
      |-- Turn 3 ------>|                |                       |
      |                 |-- LLM call --->|-- Trace (session-001) |
      |                 |                |   (linked to Turn 1+2)|-- save state
      |<-- result 3 ---|                |                       |
      |                 |                |                       |
      |-- aggregate metrics (all 3 turns for session-001) ----->|

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.observability_and_traceability.session_based_tracing
============================================================
"""

# -- Standard library --------------------------------------------------------
import sys
import json
import time
from typing import TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -- LangGraph ---------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# -- LangChain ---------------------------------------------------------------
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# -- Project imports ----------------------------------------------------------
# CONNECTION: core/ root module — get_llm() centralises LLM config.
from core.config import get_llm, get_llm_model_name
# CONNECTION: observability/ root module — build_callback_config() accepts
# session_id and user_id parameters that Langfuse uses to group related
# traces into a single session. MetricsCollector aggregates usage across
# all turns within the session.
# This script demonstrates HOW to correlate multi-turn traces in Langfuse
# via session_id — the pattern, not the callback implementation.
from observability.callbacks import build_callback_config
from observability.metrics import MetricsCollector


# ============================================================
# STAGE 4.1 -- State Definition
# ============================================================

class SessionState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_context: str
    turn_count: int
    current_query: str
    agent_response: str


# ============================================================
# STAGE 4.2 -- Node Definition
# ============================================================

# Session-level metrics collector (persists across turns)
session_metrics = MetricsCollector()


def clinical_agent_node(state: SessionState) -> dict:
    """
    Clinical agent -- responds to the current query with
    full conversation history available via add_messages.

    The key pattern here is that build_callback_config receives
    a SESSION_ID, which links all turns' traces in Langfuse.
    """
    llm = get_llm()
    current_turn = state.get("turn_count", 0) + 1
    query = state.get("current_query", "")
    patient_context = state.get("patient_context", "")

    system = SystemMessage(content=(
        f"You are a clinical decision support agent. This is turn {current_turn} "
        f"of a multi-turn consultation.\n\n"
        f"Patient context:\n{patient_context}\n\n"
        "Provide a focused, concise response (3-5 sentences). "
        "Reference previous turns if relevant."
    ))
    prompt = HumanMessage(content=query)

    # SESSION_ID in the config links all turns in Langfuse.
    # USER_ID tracks per-doctor usage.
    config = build_callback_config(
        trace_name=f"session_turn_{current_turn}",
        user_id="dr-patel-cardiology",
        session_id="session-clinical-001",
        tags=["session_demo", f"turn_{current_turn}"],
    )

    call_start = time.time()
    response = llm.invoke([system, prompt], config=config)
    latency_ms = (time.time() - call_start) * 1000

    # Record metrics for this turn
    prompt_token_estimate = len(str([system, prompt])) // 4
    completion_token_estimate = len(response.content) // 4

    session_metrics.record_llm_call(
        agent_name=f"clinical_agent_turn_{current_turn}",
        tokens_in=prompt_token_estimate,
        tokens_out=completion_token_estimate,
        model=get_llm_model_name(),
        latency_ms=latency_ms,
    )

    print(f"    | [Turn {current_turn}] Response: {len(response.content)} chars, {latency_ms:.0f}ms")

    return {
        "messages": [prompt, response],
        "agent_response": response.content,
        "turn_count": current_turn,
    }


# ============================================================
# STAGE 4.3 -- Graph Construction
# ============================================================

def build_session_graph():
    """
    Build the session-aware pipeline with MemorySaver.

    MemorySaver ensures conversation history persists across
    invoke() calls (turns). Combined with session_id in the
    Langfuse config, this gives full session traceability.
    """
    workflow = StateGraph(SessionState)
    workflow.add_node("clinical_agent", clinical_agent_node)
    workflow.add_edge(START, "clinical_agent")
    workflow.add_edge("clinical_agent", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# STAGE 4.4 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  SESSION-BASED TRACING")
    print("  Pattern: multi-turn session grouping with session_id")
    print("=" * 70)

    print("""
    Session grouping in Langfuse:

      config = build_callback_config(
          trace_name="turn_1",
          user_id="dr-patel",           <-- per-doctor analytics
          session_id="session-001",     <-- links all turns
      )

    Each invoke() creates a new Trace in Langfuse.
    All Traces with the same session_id are grouped into
    one SESSION in the Langfuse dashboard.

    Combined with MemorySaver, you get:
      - Conversation continuity (state persists across turns)
      - Session-level audit trail (who said what, when)
      - Aggregated metrics (total cost per session)
    """)

    # -- Patient context (shared across all turns) --------------------------
    patient_context = (
        "71F with CKD Stage 3a, Hypertension, CHF, Type 2 Diabetes. "
        "Presenting with dizziness and elevated potassium (K+ 5.4 mEq/L). "
        "Current medications: Lisinopril 20mg, Spironolactone 25mg, "
        "Furosemide 40mg, Metformin 500mg. eGFR 42 mL/min."
    )

    # -- Define the 3 turns of the consultation ----------------------------
    consultation_turns = [
        {
            "query": (
                "Assess this patient's elevated potassium. "
                "What is the likely cause and immediate risk?"
            ),
            "label": "Initial Assessment",
        },
        {
            "query": (
                "Given your assessment, check for drug interactions "
                "between Lisinopril and Spironolactone. Should we "
                "modify her medication regimen?"
            ),
            "label": "Drug Interaction Review",
        },
        {
            "query": (
                "Generate a brief discharge summary incorporating "
                "the assessment and medication changes discussed."
            ),
            "label": "Discharge Summary",
        },
    ]

    graph = build_session_graph()
    thread_config = {"configurable": {"thread_id": "session-demo-thread-001"}}

    print(f"    Doctor: dr-patel-cardiology")
    print(f"    Session: session-clinical-001")
    print(f"    Patient: 71F | CKD 3a, HTN, CHF, DM2")
    print(f"    Turns: {len(consultation_turns)}")
    print()

    # -- Execute each turn --------------------------------------------------
    for turn_index, turn_data in enumerate(consultation_turns):
        turn_number = turn_index + 1
        print("    " + "=" * 60)
        print(f"    TURN {turn_number}: {turn_data['label']}")
        print("    " + "-" * 60)
        print(f"    Doctor: {turn_data['query'][:100]}...")
        print()

        turn_state = {
            "messages": [],
            "patient_context": patient_context,
            "turn_count": turn_index,
            "current_query": turn_data["query"],
            "agent_response": "",
        }

        result = graph.invoke(turn_state, config=thread_config)

        # Display the agent's response
        response_text = result.get("agent_response", "")
        print(f"\n    Agent:")
        for line in response_text[:300].split("\n"):
            if line.strip():
                print(f"      {line}")

        # Show session state after this turn
        print(f"\n    Messages in history: {len(result.get('messages', []))}")
        print(f"    Turn count: {result.get('turn_count', 0)}")
        print()

    # -- Session-level metrics summary -------------------------------------
    print("    " + "=" * 60)
    print("    SESSION METRICS (aggregated across all turns)")
    print("    " + "=" * 60)

    summary = session_metrics.get_workflow_summary()
    agent_breakdown = session_metrics.get_agent_summary()

    print(f"""
    Session: session-clinical-001
    Doctor:  dr-patel-cardiology
    Turns:   {len(consultation_turns)}

    Totals:
      LLM Calls:    {summary['total_llm_calls']}
      Tokens (in):  {summary['total_tokens_in']}
      Tokens (out): {summary['total_tokens_out']}
      Total Cost:   ${summary['total_cost_usd']:.4f}
      Duration:     {summary['workflow_duration_ms']:.0f}ms
    """)

    print("    Per-Turn Breakdown:")
    print("    " + "-" * 50)
    for agent_name, agent_data in agent_breakdown.items():
        print(f"      {agent_name}:")
        print(f"        Tokens: {agent_data['total_tokens_in']} in + {agent_data['total_tokens_out']} out")
        print(f"        Cost: ${agent_data['total_cost_usd']:.4f}")
        print(f"        Latency: {agent_data.get('avg_latency_ms', 0):.0f}ms")

    # -- Checkpoint inspection (state still accessible) --------------------
    print(f"\n    CHECKPOINT STATE (via MemorySaver):")
    snapshot = graph.get_state(thread_config)
    saved_keys = list(snapshot.values.keys())
    print(f"    Saved state keys: {saved_keys}")
    print(f"    Messages preserved: {len(snapshot.values.get('messages', []))}")
    print(f"    Final turn count: {snapshot.values.get('turn_count', 0)}")

    # -- Summary -----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  SESSION-BASED TRACING SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. session_id in build_callback_config() links all turns:
         - All traces with the same session_id appear together
         - Langfuse dashboard replays the full session

      2. user_id enables per-doctor analytics:
         - Filter traces by doctor
         - Track per-doctor cost and usage

      3. MemorySaver + session_id = full audit trail:
         - Conversation history persists (MemorySaver)
         - Each turn is individually traceable (Langfuse)
         - Session-level aggregation for cost/quality analysis

      4. Session-level metrics reveal:
         - Which turn was most expensive
         - Whether follow-up turns are more efficient
         - Total cost per patient consultation

    In production:
      - session_id = patient encounter ID
      - user_id = doctor/clinician ID
      - Thread_id = conversation thread for MemorySaver
      - All three together = complete clinical audit trail

    Next: observed_clinical_pipeline.py -- all patterns combined.
    """)


if __name__ == "__main__":
    main()
