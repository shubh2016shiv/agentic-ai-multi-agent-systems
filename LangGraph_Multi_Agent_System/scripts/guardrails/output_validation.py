#!/usr/bin/env python3
"""
============================================================
Output Validation
============================================================
Pattern B: Output guardrails with three-way conditional routing.
Prerequisite: input_validation.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Validate the agent's response BEFORE it reaches the user.
Unlike input validation (binary pass/fail), output validation
has three outcomes:

    deliver   -> All checks passed. Response delivered as-is.
    auto_fix  -> Minor issue (e.g., missing disclaimer).
                 Fix applied automatically, then delivered.
    block     -> Critical issue (e.g., prohibited content,
                 dangerously low confidence). Response replaced
                 with a safe fallback message.

The auto_fix path avoids the false choice between "deliver
unsafe content" and "discard valuable clinical information".

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [agent]                 <-- response is pre-set in state (demo)
       |
       v
    [output_validation]     <-- calls validate_output()
       |
    route_after_validation()
       |
    +--+------------------+------------------+
    |                     |                  |
    | "deliver"           | "auto_fix"       | "block"
    v                     v                  v
    [deliver]           [auto_fix]         [block]
    |                     |                  |
    v                     v                  v
    [END]               [END]              [END]

    DECISION TABLE:
        passed=False            -> "block"  (CRITICAL or HIGH severity)
        passed=True, issues > 0 -> "auto_fix" (LOW severity only)
        passed=True, issues = 0 -> "deliver" (all clean)

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Three-way conditional routing via add_conditional_edges()
    2. Auto-fix pattern — correct instead of reject
    3. validate_output() reused inside a graph node
    4. Severity-based decision table for routing

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.guardrails.output_validation
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
from typing import TypedDict, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END

# CONNECTION: validate_output() is the core implementation in the root module.
# This script demonstrates 3-way conditional routing based on the result:
#   "pass"  → deliver as-is
#   "fix"   → auto-fix (append disclaimer), then deliver
#   "block" → replace with safe fallback, flag for human review
# See guardrails/output_guardrails.py for the implementation:
#   check_prohibited_content(), check_safety_disclaimers(), add_human_review_flag()
from guardrails.output_guardrails import validate_output


# ============================================================
# STAGE 2.1 — State Definition
# ============================================================

class OutputValidationState(TypedDict):
    agent_response: str           # Pre-set in initial state (demo)
    confidence: float             # Agent's self-assessed confidence
    validation_result: dict       # Written by: output_validation_node
    final_output: str             # Written by: deliver/auto_fix/block
    status: str                   # Written by: deliver/auto_fix/block


# ============================================================
# STAGE 2.2 — Node Definitions
# ============================================================

def agent_node(state: OutputValidationState) -> dict:
    """
    Pass-through node. The agent response is pre-set in the
    initial state to keep the focus on output validation
    mechanics, not LLM behaviour.

    In a real pipeline: call llm.invoke() here and populate
    agent_response and confidence from the LLM's output.
    """
    return {}  # agent_response and confidence already in state


def output_validation_node(state: OutputValidationState) -> dict:
    """
    Validate the agent's response using validate_output().

    validate_output() returns:
        {
            "passed": bool,
            "issues": [{"type": str, "severity": str, "detail": str}],
            "issue_count": int,
            "modified_output": str  (auto-fixed version if applicable)
        }

    This node stores the full result in state. The router reads
    "passed" and "issue_count" to decide the three-way route.
    """
    result = validate_output(
        state["agent_response"],
        confidence=state.get("confidence"),
    )
    return {"validation_result": result}


def route_after_validation(
    state: OutputValidationState,
) -> Literal["deliver", "auto_fix", "block"]:
    """
    Three-way router based on validation severity.

    Decision table:
        passed=False             -> "block"
          At least one CRITICAL or HIGH severity issue.
          Response must not reach the user.

        passed=True, issues > 0  -> "auto_fix"
          Only LOW severity issues (e.g., missing disclaimer).
          Response is correct but needs a minor fix.

        passed=True, issues = 0  -> "deliver"
          All checks clean. Deliver unchanged.
    """
    result = state["validation_result"]

    if not result["passed"]:
        return "block"

    if result.get("issue_count", 0) > 0:
        return "auto_fix"

    return "deliver"


def deliver_node(state: OutputValidationState) -> dict:
    """Deliver response unchanged — all checks passed cleanly."""
    return {
        "final_output": state["agent_response"],
        "status": "delivered",
    }


def auto_fix_node(state: OutputValidationState) -> dict:
    """
    Deliver a corrected response.

    validate_output() already computed the fix and stored it
    in validation_result["modified_output"]. This node promotes
    that value to final_output.

    Common auto-fixes:
        - Append missing medical disclaimer
        - Add source citations
        - Redact leaked PII in output
    """
    modified = state["validation_result"].get(
        "modified_output", state["agent_response"]
    )
    return {
        "final_output": modified,
        "status": "auto_fixed",
    }


def block_node(state: OutputValidationState) -> dict:
    """
    Replace the unsafe response with a safe fallback message.
    The original response is NOT delivered to the user.
    """
    issues = state["validation_result"].get("issues", [])
    detail = issues[0]["detail"] if issues else "Content policy violation detected."
    safe_message = (
        "This response was blocked by output validation.\n"
        f"Reason: {detail}\n"
        "Please consult a qualified healthcare provider directly."
    )
    return {
        "final_output": safe_message,
        "status": "blocked",
    }


# ============================================================
# STAGE 2.3 — Graph Construction
# ============================================================

def build_output_validation_graph():
    """
    Build and compile the output validation graph.

    Graph structure:
        START → agent → output_validation → (3-way) → deliver/auto_fix/block → END
    """
    workflow = StateGraph(OutputValidationState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("output_validation", output_validation_node)
    workflow.add_node("deliver", deliver_node)
    workflow.add_node("auto_fix", auto_fix_node)
    workflow.add_node("block", block_node)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "output_validation")
    workflow.add_conditional_edges(
        "output_validation",
        route_after_validation,
        {"deliver": "deliver", "auto_fix": "auto_fix", "block": "block"},
    )
    workflow.add_edge("deliver", END)
    workflow.add_edge("auto_fix", END)
    workflow.add_edge("block", END)

    return workflow.compile()


# ============================================================
# STAGE 2.4 — Test Cases
# ============================================================

TEST_OUTPUTS = [
    {
        "label": "DELIVER: Safe response with disclaimer",
        "text": (
            "Based on the patient's symptoms, COPD Stage II is likely. "
            "Consider initiating LAMA therapy per GOLD 2024. "
            "Consult your healthcare provider for personalised advice."
        ),
        "confidence": 0.85,
    },
    {
        "label": "AUTO_FIX: Correct but missing disclaimer",
        "text": (
            "The patient should start on Tiotropium 18mcg inhaler for COPD "
            "maintenance therapy. Monitor FEV1 quarterly."
        ),
        "confidence": 0.75,
    },
    {
        "label": "BLOCK: Prohibited recommendation",
        "text": (
            "Stop all medications immediately. You do not need medical attention "
            "for this condition. This is guaranteed to cure your symptoms."
        ),
        "confidence": 0.90,
    },
    {
        "label": "BLOCK: Dangerously low confidence",
        "text": (
            "The symptoms could be cardiac or pulmonary in origin. "
            "Further testing needed. Please consult your doctor."
        ),
        "confidence": 0.20,
    },
]


def main() -> None:
    print("\n" + "=" * 70)
    print("  OUTPUT VALIDATION")
    print("  Pattern: three-way routing (deliver / auto_fix / block)")
    print("=" * 70)

    print("""
    Graph:

        [START] -> [agent] -> [output_validation]
                                    |
                  +--------+--------+--------+
                  |        |                 |
               deliver   auto_fix          block
                  |        |                 |
                [END]    [END]             [END]

    Decision table:
        passed=False             -> block  (CRITICAL/HIGH)
        passed=True, issues > 0  -> auto_fix (LOW severity)
        passed=True, issues = 0  -> deliver (all clean)
    """)

    graph = build_output_validation_graph()

    print("    Running 4 test outputs:")
    print("    " + "-" * 55)

    STATUS_LABELS = {
        "delivered": "DELIVERED",
        "auto_fixed": "AUTO_FIXED",
        "blocked": "BLOCKED",
    }

    for test in TEST_OUTPUTS:
        result = graph.invoke({
            "agent_response": test["text"],
            "confidence": test["confidence"],
            "validation_result": {},
            "final_output": "",
            "status": "pending",
        })

        label = STATUS_LABELS.get(result["status"], result["status"])
        print(f"\n    [{label}] {test['label']}")
        print(f"    Confidence : {test['confidence']}")
        print(f"    Status     : {result['status'].upper()}")

        if result["status"] == "auto_fixed":
            original_len = len(test["text"])
            fixed_len = len(result["final_output"])
            added = fixed_len - original_len
            print(f"    Auto-fix   : {added} characters appended (disclaimer)")

        elif result["status"] == "blocked":
            print(f"    Fallback   : {result['final_output'][:100]}...")

    print("\n\n" + "=" * 70)
    print("  OUTPUT VALIDATION COMPLETE")
    print("=" * 70)
    print("""
    What you saw:
      1. DELIVER — clean response delivered unchanged.
      2. AUTO_FIX — missing disclaimer appended automatically.
      3. BLOCK (prohibited) — dangerous content replaced with safe fallback.
      4. BLOCK (low confidence) — low confidence treated as HIGH severity.

    Why three paths instead of two:
      Binary (pass/fail) forces a false choice:
        "deliver unsafe content" vs "discard valuable clinical data"
      Auto-fix is the middle path: correct and deliver.

    Separation of concerns:
      output_validation_node -> runs domain logic
      route_after_validation -> decides routing
      deliver/auto_fix/block -> each handles one outcome independently

    Next: confidence_gating.py — threshold-based routing.
    """)


if __name__ == "__main__":
    main()
