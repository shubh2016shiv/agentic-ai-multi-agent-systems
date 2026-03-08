#!/usr/bin/env python3
"""
============================================================
Tool Call Confirmation
============================================================
Pattern B: Pause before executing a tool call so a human
can approve, modify, or reject the tool invocation.
Prerequisite: basic_approval.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
When an LLM agent proposes a tool call, pause the pipeline
so a human can review what the agent wants to do BEFORE
it happens.

This is critical for tools with side effects:
    - Database writes (prescriptions, orders)
    - External API calls (lab requests, referrals)
    - Financial transactions
    - Any irreversible action

The pattern:
    1. Agent proposes a tool call (via bind_tools)
    2. Pipeline pauses BEFORE executing the tool
    3. Human reviews the tool name and arguments
    4. Human approves (execute), modifies (edit args), or rejects (skip)
    5. Pipeline continues with the human's decision

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [agent]              <-- real LLM call, proposes tool call
       |
       v
    [confirm_tool]       <-- interrupt() with tool call details
       |
    route_after_confirm()
       |
    +--+---------+-----------+
    |             |           |
    | "execute"   | "skip"    | (future: "modify")
    v             v           v
    [execute]    [skip]      [modify]
    |             |           |
    v             v           v
    [respond]    [respond]   [respond]
       |
       v
    [END]

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Intercepting tool calls before execution
    2. Structured resume values (dict, not just bool)
    3. Conditional routing based on human decision
    4. Tool call inspection (name, args, id)

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.HITL.tool_call_confirmation
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import json
from typing import TypedDict, Literal, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage

# ── Project imports ─────────────────────────────────────────────────────────
from core.config import get_llm
from core.models import PatientCase
from tools import analyze_symptoms, assess_patient_risk
from observability.callbacks import build_callback_config

# CONNECTION: Tool confirmation uses hitl.primitives for the interrupt payload
# and resume parsing. build_tool_payload() creates the structured interrupt
# payload showing tool_name, tool_args, and tool_id. parse_resume_action()
# normalises the dict resume value {"action": "execute"} into a standard dict.
# See hitl/primitives.py for the concept explanation.
from hitl.primitives import build_tool_payload, parse_resume_action
from hitl.run_cycle import run_hitl_cycle


# ============================================================
# STAGE 2.1 — State Definition
# ============================================================

class ToolConfirmState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    proposed_tool_call: dict    # Written by: agent_node (tool call details)
    human_decision: str         # Written by: confirm_tool_node
    tool_result: str            # Written by: execute_tool_node or skip_node
    final_response: str         # Written by: respond_node
    status: str                 # "executed" | "skipped"


# ============================================================
# STAGE 2.2 — Node Definitions
# ============================================================

def agent_node(state: ToolConfirmState) -> dict:
    """
    Clinical agent — real LLM call that proposes a tool call.

    The agent is prompted to use a tool. We capture the tool
    call proposal WITHOUT executing it, so the human can
    review it first.
    """
    llm = get_llm()
    tools = [analyze_symptoms, assess_patient_risk]
    agent_llm = llm.bind_tools(tools)

    patient = state["patient_case"]
    system = SystemMessage(content=(
        "You are a clinical triage specialist. "
        "Use your tools to analyze the patient. "
        "Call analyze_symptoms or assess_patient_risk with the patient data."
    ))
    prompt = HumanMessage(content=f"""Evaluate this patient:
Age: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
Vitals: {json.dumps(patient.get('vitals', {}))}""")

    config = build_callback_config(trace_name="tool_confirm_agent")
    response = agent_llm.invoke([system, prompt], config=config)

    # Capture the proposed tool call (if any)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tc = response.tool_calls[0]
        proposed = {
            "name": tc["name"],
            "args": tc["args"],
            "id": tc.get("id", "unknown"),
        }
        print(f"    | [Agent] Proposed tool call: {tc['name']}")
        print(f"    |   Args: {json.dumps(tc['args'])[:100]}...")
        return {
            "messages": [response],
            "proposed_tool_call": proposed,
        }

    # No tool call — agent responded directly
    print(f"    | [Agent] No tool call proposed (direct response)")
    return {
        "messages": [response],
        "proposed_tool_call": {},
        "final_response": response.content,
        "status": "direct_response",
    }


def confirm_tool_node(state: ToolConfirmState) -> dict:
    """
    Pause for human to review the proposed tool call.

    The interrupt payload shows the human:
        - Which tool the agent wants to call
        - What arguments it wants to pass
        - The tool call ID

    The human responds with Command(resume={"action": "..."})
    """
    proposed = state.get("proposed_tool_call", {})

    if not proposed:
        # No tool call to confirm — skip this node
        return {"human_decision": "no_tool_call"}

    print(f"    | [Confirm] Requesting human approval for tool call...")
    print(f"    |   Tool: {proposed.get('name')}")
    print(f"    |   Args: {json.dumps(proposed.get('args', {}))[:100]}")

    # ── interrupt() PAUSES HERE ──────────────────────────────────────
    # CONNECTION: build_tool_payload() from hitl.primitives creates the
    # standardised interrupt payload for tool-call confirmation. It includes
    # tool_name, tool_args, tool_id, and options=["execute", "skip"].
    decision = interrupt(build_tool_payload(
        tool_name=proposed.get("name", "unknown"),
        tool_args=proposed.get("args", {}),
        tool_id=proposed.get("id", "unknown"),
    ))

    # ── Resumes here with the human's decision ───────────────────────
    # CONNECTION: parse_resume_action() normalises the resume value.
    # The resume value here is a dict: {"action": "execute"} or {"action": "skip"}
    parsed = parse_resume_action(decision, default_action="skip")
    action = parsed["action"]
    print(f"    | [Confirm] Human decision: {action.upper()}")
    return {"human_decision": action}


def route_after_confirm(state: ToolConfirmState) -> Literal["execute_tool", "skip_tool", "respond"]:
    """Route based on the human's decision about the tool call."""
    decision = state.get("human_decision", "skip")
    if decision == "execute":
        return "execute_tool"
    if decision == "no_tool_call":
        return "respond"
    return "skip_tool"


def execute_tool_node(state: ToolConfirmState) -> dict:
    """Execute the approved tool call and store the result."""
    proposed = state["proposed_tool_call"]
    tools = [analyze_symptoms, assess_patient_risk]
    tool_node = ToolNode(tools)

    # Reconstruct the tool call message for ToolNode
    last_ai_msg = None
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            last_ai_msg = msg
            break

    if last_ai_msg:
        result = tool_node.invoke({"messages": [last_ai_msg]})
        tool_messages = result.get("messages", [])
        tool_content = tool_messages[0].content if tool_messages else "Tool returned no result"
        print(f"    | [Execute] Tool executed: {proposed.get('name')}")
        print(f"    |   Result: {str(tool_content)[:100]}...")
        return {
            "messages": tool_messages,
            "tool_result": str(tool_content),
        }

    return {"tool_result": "Could not execute tool (message not found)"}


def skip_tool_node(state: ToolConfirmState) -> dict:
    """Human rejected the tool call — skip execution."""
    proposed = state["proposed_tool_call"]
    print(f"    | [Skip] Tool call skipped: {proposed.get('name')}")
    return {
        "tool_result": f"Tool call '{proposed.get('name')}' was skipped by human reviewer.",
    }


def respond_node(state: ToolConfirmState) -> dict:
    """
    Generate final response after tool execution or skip.

    If the tool was executed, incorporate its result.
    If skipped, acknowledge the skip.
    """
    if state.get("status") == "direct_response":
        return {}

    tool_result = state.get("tool_result", "")
    decision = state.get("human_decision", "")

    if decision == "execute":
        output = (
            f"Tool executed successfully.\n"
            f"Result: {tool_result[:300]}"
        )
        status = "executed"
    elif decision == "no_tool_call":
        output = state.get("final_response", "Agent responded without tool calls.")
        status = "direct_response"
    else:
        output = (
            f"Tool call was skipped by reviewer.\n"
            f"Note: {tool_result}"
        )
        status = "skipped"

    return {"final_response": output, "status": status}


# ============================================================
# STAGE 2.3 — Graph Construction
# ============================================================

def build_confirmation_graph():
    """
    Build the tool call confirmation graph.

    Graph: START -> agent -> confirm_tool -> (execute/skip) -> respond -> END
    """
    workflow = StateGraph(ToolConfirmState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("confirm_tool", confirm_tool_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("skip_tool", skip_tool_node)
    workflow.add_node("respond", respond_node)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "confirm_tool")
    workflow.add_conditional_edges(
        "confirm_tool",
        route_after_confirm,
        {"execute_tool": "execute_tool", "skip_tool": "skip_tool", "respond": "respond"},
    )
    workflow.add_edge("execute_tool", "respond")
    workflow.add_edge("skip_tool", "respond")
    workflow.add_edge("respond", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# STAGE 2.4 — Test Cases
# ============================================================

TEST_PATIENT = PatientCase(
    patient_id="PT-TC-001",
    age=58, sex="M",
    chief_complaint="Persistent cough and dyspnea for 3 weeks",
    symptoms=["cough", "dyspnea", "wheezing"],
    medical_history=["COPD Stage II"],
    current_medications=["Tiotropium 18mcg"],
    allergies=[],
    lab_results={"FEV1": "58% predicted"},
    vitals={"BP": "138/85", "HR": "92", "SpO2": "93%"},
)


def make_state() -> ToolConfirmState:
    return {
        "messages": [],
        "patient_case": TEST_PATIENT.model_dump(),
        "proposed_tool_call": {},
        "human_decision": "",
        "tool_result": "",
        "final_response": "",
        "status": "pending",
    }


def run_tool_confirm_cycle(graph, thread_id: str, human_decision: str) -> dict:
    """
    Run a complete tool confirmation cycle.

    CONNECTION: Delegates to run_hitl_cycle() from hitl.run_cycle.
    The resume value for Pattern B is a dict {"action": "execute"|"skip"}.
    """
    # CONNECTION: run_hitl_cycle() handles invoke → pause → resume.
    # The resume_value here is a dict because Pattern B uses dict resumes.
    final = run_hitl_cycle(
        graph=graph,
        thread_id=thread_id,
        initial_state=make_state(),
        resume_value={"action": human_decision},
        verbose=True,
    )

    print(f"\n    STATUS: {final.get('status', 'unknown').upper()}")
    print(f"    Output: {final.get('final_response', '')[:150]}...")
    return final


def main() -> None:
    print("\n" + "=" * 70)
    print("  TOOL CALL CONFIRMATION")
    print("  Pattern: interrupt before tool execution")
    print("=" * 70)

    print("""
    When an agent proposes a tool call:

      [agent] -> proposes tool call
                    |
      [confirm_tool] -> interrupt({tool_name, tool_args})
                    |
                 PAUSED — human reviews
                    |
      Command(resume={"action": "execute"})  <- approve
      Command(resume={"action": "skip"})     <- reject
                    |
      [execute_tool] or [skip_tool] -> [respond] -> [END]

    Why this matters:
      Tools with side effects (database writes, API calls,
      prescriptions) should never execute without human review.
    """)

    graph = build_confirmation_graph()
    print("    Graph compiled with MemorySaver.\n")

    # ── Test 1: Approve tool call ─────────────────────────────────────
    print("=" * 70)
    print("  TEST 1: Human APPROVES tool call")
    print("=" * 70)
    r1 = run_tool_confirm_cycle(graph, thread_id="tc-approve-001", human_decision="execute")

    # ── Test 2: Reject tool call ──────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 2: Human REJECTS tool call")
    print("=" * 70)
    r2 = run_tool_confirm_cycle(graph, thread_id="tc-reject-001", human_decision="skip")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  TOOL CALL CONFIRMATION COMPLETE")
    print("=" * 70)
    print(f"""
    Results:
      Test 1 (approve): {r1.get('status', 'unknown').upper()}
      Test 2 (reject):  {r2.get('status', 'unknown').upper()}

    Key differences from basic_approval:
      1. Resume value is a dict, not just True/False.
      2. The interrupt payload includes tool-specific details.
      3. Conditional routing after human decision.
      4. Real LLM call generating the tool proposal.

    When to use this pattern:
      - Database writes (prescriptions, orders)
      - External API calls (lab requests, referrals)
      - Financial transactions
      - Any irreversible action

    Next: edit_before_approve.py — modify response before delivery.
    """)


if __name__ == "__main__":
    main()
