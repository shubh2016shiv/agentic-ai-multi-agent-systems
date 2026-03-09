#!/usr/bin/env python3
"""
============================================================
Tool Error Handling
============================================================
Prerequisite: dynamic_tool_selection.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Handle tool failures gracefully: when a tool raises an
exception, the error is captured and fed back to the LLM
so it can self-correct (retry with different args, try a
different tool, or produce a response without tools).

This script shows three patterns:
    1. ToolNode's built-in error handling (handle_tool_errors=True)
    2. Manual try/except inside a node with fallback logic
    3. Combining with a retry counter to prevent infinite loops

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    PATTERN 1 — ToolNode handles errors automatically:

        [agent] -> [tools (handle_tool_errors=True)]
           ^              |
           |      (error) -> converts to ToolMessage(error=...)
           |              |
           +--- LLM sees error, retries or gives up

    PATTERN 2 — Manual error handling:

        [agent]
           |  tool_call -> try/except
           |  success?  -> feed result back
           |  failure?  -> increment retry_count
           |              -> if retries < max: retry
           |              -> if retries >= max: respond without tool

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. handle_tool_errors=True on ToolNode
    2. Manual try/except with error feedback to the LLM
    3. Retry counter to limit error loops
    4. Fallback behaviour when tools fail

------------------------------------------------------------
WHEN TO USE
------------------------------------------------------------
    Use tool_error_handling patterns in any production pipeline
    where tools can fail (network timeouts, invalid args, API errors).

    Pattern 1 (handle_tool_errors=True): use when you want ToolNode
    to capture exceptions and feed them to the LLM for self-correction.

    Pattern 2 (manual try/except): use when you need explicit fallback
    logic — retry with different args, skip the tool, or respond without it.

    When NOT to use:
    - In demos/exploratory scripts where error handling adds noise

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.tools.tool_error_handling
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import os
import json
from typing import TypedDict, Annotated, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ── Project imports ─────────────────────────────────────────────────────────
# CONNECTION: core/ root module — get_llm() centralises LLM config.
# PatientCase is the canonical domain model used in test scenarios.
from core.config import get_llm
from core.models import PatientCase
# CONNECTION: tools/ root module — real tools from the component layer.
# This script also defines flaky_tools (for error simulation) locally.
# analyze_symptoms and assess_patient_risk are the healthy tools used in
# Pattern 2 as fallback alternatives when flaky tools fail.
from tools import analyze_symptoms, assess_patient_risk
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse trace_name and tags to every LLM call automatically.
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 10.1 — Flaky Tools (for demonstration)
# ============================================================
# These tools simulate real-world failures: timeout, bad input,
# intermittent service errors.

call_counter = {"flaky_lookup": 0, "strict_dosage": 0}


@tool
def flaky_drug_lookup(drug_name: str) -> str:
    """
    Look up drug information. This tool is unreliable —
    it fails on the first call but succeeds on retry.

    Args:
        drug_name: Name of the drug to look up.
    """
    call_counter["flaky_lookup"] += 1
    count = call_counter["flaky_lookup"]

    if count % 2 == 1:
        # Fail on odd calls (1st, 3rd, 5th...)
        raise ConnectionError(
            f"Drug database temporarily unavailable (attempt {count}). "
            "Please retry."
        )

    # Succeed on even calls
    return json.dumps({
        "drug": drug_name,
        "class": "ACE Inhibitor" if "lisinopril" in drug_name.lower() else "Unknown",
        "common_side_effects": ["dizziness", "cough", "hyperkalemia"],
        "renal_adjustment": "Reduce dose if eGFR < 30 mL/min",
    })


@tool
def strict_dosage_calculator(
    drug_name: str,
    current_dose_mg: float,
    egfr: float,
) -> str:
    """
    Calculate dosage adjustment based on renal function.
    Requires EXACT numeric inputs — no units, no text.

    Args:
        drug_name: Name of the drug.
        current_dose_mg: Current dose in mg (numeric only).
        egfr: eGFR value in mL/min (numeric only).
    """
    call_counter["strict_dosage"] += 1

    # Validate inputs strictly
    if not isinstance(current_dose_mg, (int, float)) or current_dose_mg <= 0:
        raise ValueError(
            f"current_dose_mg must be a positive number, got: {current_dose_mg}"
        )
    if not isinstance(egfr, (int, float)) or egfr <= 0:
        raise ValueError(
            f"egfr must be a positive number, got: {egfr}"
        )

    adjustment = 1.0 if egfr >= 60 else 0.75 if egfr >= 30 else 0.5
    new_dose = current_dose_mg * adjustment

    return json.dumps({
        "drug": drug_name,
        "original_dose_mg": current_dose_mg,
        "egfr": egfr,
        "adjustment_factor": adjustment,
        "recommended_dose_mg": new_dose,
        "note": f"{'No adjustment needed' if adjustment == 1.0 else f'Reduce to {new_dose:.0f}mg due to renal impairment'}",
    })


# ============================================================
# STAGE 10.2 — Pattern 1: ToolNode with handle_tool_errors
# ============================================================

class ErrorDemoState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_response: str
    error_count: int


def build_pattern_1():
    """
    Build a graph where ToolNode captures errors automatically.

    handle_tool_errors=True makes ToolNode catch exceptions
    and convert them to ToolMessage objects with the error text.
    The LLM sees the error and can retry or adapt.
    """
    llm = get_llm()
    tools = [flaky_drug_lookup, strict_dosage_calculator, analyze_symptoms]
    agent_llm = llm.bind_tools(tools)

    def agent_node(state: ErrorDemoState) -> dict:
        config = build_callback_config(trace_name="tool_error_p1")
        response = agent_llm.invoke(state["messages"], config=config)

        if hasattr(response, "tool_calls") and response.tool_calls:
            calls = [tc["name"] for tc in response.tool_calls]
            print(f"    | [P1 Agent] Tool calls: {calls}")
        else:
            print(f"    | [P1 Agent] Final response ({len(response.content)} chars)")

        return {"messages": [response]}

    def should_use_tools(state: ErrorDemoState) -> Literal["tools", "end"]:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "end"

    # ToolNode with error handling enabled
    tool_node = ToolNode(
        tools,
        handle_tool_errors=True,  # <-- This is the key setting
    )

    workflow = StateGraph(ErrorDemoState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_use_tools,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ============================================================
# STAGE 10.3 — Pattern 2: Manual Error Handling
# ============================================================

def build_pattern_2():
    """
    Build a graph with manual try/except error handling
    inside the agent node. Includes a retry counter.
    """
    llm = get_llm()
    tools = [flaky_drug_lookup, strict_dosage_calculator, analyze_symptoms]
    agent_llm = llm.bind_tools(tools)
    max_retries = 3

    def agent_node(state: ErrorDemoState) -> dict:
        config = build_callback_config(trace_name="tool_error_p2")
        messages = list(state["messages"])
        response = agent_llm.invoke(messages, config=config)
        retries = state.get("error_count", 0)

        while hasattr(response, "tool_calls") and response.tool_calls:
            tool_results = []
            had_error = False

            for tc in response.tool_calls:
                print(f"    | [P2 Agent] Calling: {tc['name']}")

                # Find the matching tool function
                tool_fn = next(
                    (t for t in tools if t.name == tc["name"]),
                    None
                )

                if tool_fn is None:
                    error_msg = f"Tool '{tc['name']}' not found."
                    print(f"    | [P2 Agent] ERROR: {error_msg}")
                    tool_results.append(ToolMessage(
                        content=f"Error: {error_msg}",
                        tool_call_id=tc["id"],
                    ))
                    had_error = True
                    continue

                try:
                    result = tool_fn.invoke(tc["args"])
                    print(f"    | [P2 Agent] Success: {tc['name']}")
                    tool_results.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    ))
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    print(f"    | [P2 Agent] ERROR: {error_msg}")
                    tool_results.append(ToolMessage(
                        content=f"Tool error: {error_msg}. You may retry with different arguments.",
                        tool_call_id=tc["id"],
                    ))
                    had_error = True
                    retries += 1

            messages.extend([response] + tool_results)

            if had_error and retries >= max_retries:
                print(f"    | [P2 Agent] Max retries ({max_retries}) reached. Asking LLM to respond without tool.")
                messages.append(HumanMessage(content=(
                    "Tool calls have failed repeatedly. Please provide "
                    "your best assessment based on available information."
                )))

            response = agent_llm.invoke(messages, config=config)

        print(f"    | [P2 Agent] Final response ({len(response.content)} chars)")

        return {
            "messages": [response],
            "agent_response": response.content,
            "error_count": retries,
        }

    workflow = StateGraph(ErrorDemoState)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


# ============================================================
# STAGE 10.4 — Execute
# ============================================================

PATIENT = PatientCase(
    patient_id="PT-ERR-001",
    age=71, sex="F",
    chief_complaint="Dizziness with elevated K+",
    symptoms=["dizziness", "fatigue"],
    medical_history=["CKD Stage 3a", "Hypertension"],
    current_medications=["Lisinopril 20mg"],
    allergies=[],
    lab_results={"K+": "5.4 mEq/L", "eGFR": "42 mL/min"},
    vitals={"BP": "105/65"},
)

PROMPT = f"""Evaluate this patient and look up drug information for Lisinopril:
Patient: {PATIENT.age}y {PATIENT.sex} — {PATIENT.chief_complaint}
Medications: {', '.join(PATIENT.current_medications)}
Labs: K+={PATIENT.lab_results['K+']}, eGFR={PATIENT.lab_results['eGFR']}

Steps:
1. Look up Lisinopril drug information
2. Calculate dosage adjustment (current_dose_mg=20, egfr=42)
3. Analyze symptoms
4. Provide your assessment"""


def main() -> None:
    print("\n" + "=" * 70)
    print("  TOOL ERROR HANDLING")
    print("  Pattern: graceful recovery from tool failures")
    print("=" * 70)

    print("""
    Two error handling patterns:

    Pattern 1: ToolNode(handle_tool_errors=True)
      -> ToolNode catches exceptions automatically
      -> Converts them to ToolMessage(content="Error: ...")
      -> LLM sees the error and can self-correct

    Pattern 2: Manual try/except with retry counter
      -> Full control over error handling
      -> Can limit retries and force fallback
      -> Can log errors differently per tool

    Flaky tools used for demonstration:
      flaky_drug_lookup     : fails on 1st call, succeeds on 2nd
      strict_dosage_calculator : requires exact numeric inputs
    """)

    # ── Pattern 1 ───────────────────────────────────────────────────────
    print("=" * 70)
    print("  PATTERN 1: ToolNode with handle_tool_errors=True")
    print("=" * 70)

    # Reset counters
    call_counter["flaky_lookup"] = 0
    call_counter["strict_dosage"] = 0

    graph_p1 = build_pattern_1()
    initial_state: ErrorDemoState = {
        "messages": [
            SystemMessage(content="Evaluate the patient. Use your tools."),
            HumanMessage(content=PROMPT),
        ],
        "agent_response": "",
        "error_count": 0,
    }

    result_p1 = graph_p1.invoke(initial_state)
    print(f"\n    Tool call attempts: flaky={call_counter['flaky_lookup']}, strict={call_counter['strict_dosage']}")

    # ── Pattern 2 ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  PATTERN 2: Manual try/except with retry counter")
    print("=" * 70)

    # Reset counters
    call_counter["flaky_lookup"] = 0
    call_counter["strict_dosage"] = 0

    graph_p2 = build_pattern_2()
    result_p2 = graph_p2.invoke(initial_state)
    print(f"\n    Tool call attempts: flaky={call_counter['flaky_lookup']}, strict={call_counter['strict_dosage']}")
    print(f"    Total errors caught: {result_p2.get('error_count', 0)}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  TOOL ERROR HANDLING COMPLETE")
    print("=" * 70)
    print("""
    What you saw:
      Pattern 1: ToolNode caught errors and fed them back to the LLM.
                 The LLM saw the error message and retried or adapted.
      Pattern 2: Manual try/except gave full control over retries.
                 After max retries, the LLM was asked to respond without tools.

    When to use which:
      Pattern 1: simpler setup, good for most cases.
      Pattern 2: when you need retry limits, error logging,
                 per-tool fallback, or custom error messages.

    Production tips:
      - Always set handle_tool_errors=True on production ToolNodes.
      - Implement retry counters to prevent infinite error loops.
      - Log tool failures for monitoring (Langfuse traces help here).
      - Consider fallback tools for critical functionality.

    This completes the tool pattern series.
    """)


if __name__ == "__main__":
    main()
