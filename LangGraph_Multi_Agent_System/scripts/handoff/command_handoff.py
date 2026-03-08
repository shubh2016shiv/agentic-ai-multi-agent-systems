#!/usr/bin/env python3
"""
============================================================
Command-Based Handoff
============================================================
Prerequisite: conditional_routing.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Let the LLM decide which agent runs next by calling a
"transfer tool". The tool returns a Command(goto=, update=)
that rewires the graph at runtime.

In linear_pipeline.py, the developer decided the order.
In conditional_routing.py, a Python function decided.
Here, the LLM itself decides — highest flexibility, but
requires depth guards to prevent runaway chains.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    [START]
       |
       v
    [triage]
       |
       | LLM calls transfer_to_pharmacology(reason=..., findings=[...])
       |   -> tool returns Command(goto="pharmacology", update={...})
       v
    [pharmacology]
       |
       | LLM calls transfer_to_report(summary=...)
       |   -> tool returns Command(goto="report", update={...})
       v
    [report]
       |
       v
     [END]

    Routing:  Command objects returned by handoff tools.
    Who decides: THE LLM (via tool selection).
    Token cost: same as a regular tool call.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Command(goto=, update=) — atomic routing + state update
    2. Handoff tools — @tool functions that return Command
    3. Tool factory pattern — create_transfer_tools()
    4. Mixing domain tools and handoff tools in one bind_tools() call
    5. Depth guard with Command-based routing

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.handoff.command_handoff
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import json
from typing import TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

# ── Project imports ─────────────────────────────────────────────────────────
from core.config import get_llm
from core.models import PatientCase
from tools import (
    analyze_symptoms,
    assess_patient_risk,
    check_drug_interactions,
    lookup_drug_info,
    calculate_dosage_adjustment,
)
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 3.1 — State Definition
# ============================================================

class CommandState(TypedDict):
    """
    State for Command-based handoff.

    handoff_context is a dict written by transfer tools
    via Command(update={"handoff_context": {...}}).
    """
    messages: Annotated[list, add_messages]
    patient_case: dict
    handoff_context: dict
    current_agent: str
    handoff_history: list[str]
    handoff_depth: int
    max_handoff_depth: int
    final_report: str


# ============================================================
# STAGE 3.2 — Transfer Tool Factory
# ============================================================
# Transfer tools are @tool functions that return Command objects.
# They are what the LLM calls to hand off control.
#
# The factory pattern is used because the tools need access to
# the current handoff_depth (from state) to enforce depth limits.
# The factory creates fresh tools with the current depth value
# captured via closure.

def create_transfer_tools(current_depth: int, max_depth: int):
    """
    Create handoff tools bound to the current depth.

    Each tool returns a Command that:
      1. Sets goto= to redirect execution to the target node.
      2. Sets update= to merge handoff data into shared state.

    Args:
        current_depth: Current handoff depth from state.
        max_depth: Maximum allowed depth.

    Returns:
        List of @tool functions.
    """

    @tool
    def transfer_to_pharmacology(
        reason: str,
        urgent_findings: list[str],
    ) -> Command:
        """
        Transfer this case to the pharmacology agent for drug
        interaction review. Call this when you identify:
        - Dangerous drug combinations
        - Renal dosing concerns (low eGFR)
        - Elevated potassium with potassium-raising agents

        Args:
            reason: Why you are initiating this handoff.
            urgent_findings: The 3-5 most critical findings
                             the pharmacologist must address.
        """
        if current_depth >= max_depth:
            # Depth guard: refuse the handoff if at limit
            return Command(
                goto="report",
                update={
                    "handoff_context": {
                        "blocked": True,
                        "reason": f"Depth limit ({max_depth}) reached.",
                    },
                    "handoff_depth": current_depth + 1,
                    "handoff_history": ["depth_guard_redirect"],
                },
            )

        return Command(
            goto="pharmacology",
            update={
                "handoff_context": {
                    "from_agent": "triage",
                    "to_agent": "pharmacology",
                    "reason": reason,
                    "urgent_findings": urgent_findings,
                },
                "current_agent": "triage",
                "handoff_depth": current_depth + 1,
                "handoff_history": ["triage"],
            },
        )

    @tool
    def transfer_to_report(summary: str) -> Command:
        """
        Transfer to the report generator to produce the final
        clinical summary. Call this when your analysis is complete.

        Args:
            summary: Brief summary of your findings for the report.
        """
        return Command(
            goto="report",
            update={
                "handoff_context": {
                    "from_agent": "pharmacology",
                    "to_agent": "report",
                    "summary": summary,
                },
                "current_agent": "pharmacology",
                "handoff_depth": current_depth + 1,
                "handoff_history": ["pharmacology"],
            },
        )

    return [transfer_to_pharmacology, transfer_to_report]


# ============================================================
# STAGE 3.3 — Build the Graph
# ============================================================

def build_command_pipeline():
    """Build a graph where routing is driven by Command objects."""
    llm = get_llm()

    # Domain tools
    triage_domain_tools = [analyze_symptoms, assess_patient_risk]
    pharma_domain_tools = [check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment]

    patient = PatientCase(
        patient_id="PT-CMD-001",
        age=71, sex="F",
        chief_complaint="Dizziness and fatigue after medication change",
        symptoms=["dizziness", "fatigue", "headache", "bilateral ankle edema"],
        medical_history=["Hypertension", "CKD Stage 3a", "Type 2 Diabetes"],
        current_medications=[
            "Lisinopril 20mg daily",
            "Spironolactone 25mg daily",
            "Metformin 1000mg BID",
            "Amlodipine 10mg daily",
        ],
        allergies=["Sulfa drugs"],
        lab_results={"eGFR": "42 mL/min", "K+": "5.4 mEq/L", "Cr": "1.6 mg/dL"},
        vitals={"BP": "105/65", "HR": "88", "SpO2": "95%"},
    )

    # ── Node: triage ────────────────────────────────────────────────────
    def triage_node(state: CommandState) -> Command:
        """
        Triage agent with domain tools AND transfer tools.

        The LLM sees both tool sets. After using domain tools to
        assess the patient, it calls transfer_to_pharmacology()
        to hand off control. The transfer tool returns a Command.
        """
        print("\n    [STAGE 3.3] TRIAGE AGENT (with handoff tools)")
        print("    " + "-" * 50)

        # Create transfer tools with current depth
        transfer_tools = create_transfer_tools(
            state["handoff_depth"], state["max_handoff_depth"]
        )
        all_tools = triage_domain_tools + transfer_tools
        triage_llm = llm.bind_tools(all_tools)

        system_msg = SystemMessage(content=(
            "You are a triage specialist. Evaluate the patient using your "
            "domain tools (analyze_symptoms, assess_patient_risk). Then "
            "decide: if the patient needs medication review, call "
            "transfer_to_pharmacology with a reason and findings. "
            "Do NOT prescribe or adjust medications yourself."
        ))

        user_msg = HumanMessage(content=f"""Evaluate this patient:

Patient: {patient.age}y {patient.sex}
Chief Complaint: {patient.chief_complaint}
Symptoms: {', '.join(patient.symptoms)}
Medications: {', '.join(patient.current_medications)}
Labs: K+ = {patient.lab_results.get('K+')}, eGFR = {patient.lab_results.get('eGFR')}
Vitals: BP = {patient.vitals.get('BP')}, HR = {patient.vitals.get('HR')}

Use your tools to assess, then transfer to the appropriate specialist.""")

        config = build_callback_config(trace_name="handoff_cmd_triage")
        messages = [system_msg, user_msg]
        response = triage_llm.invoke(messages, config=config)

        # Process tool calls — domain tools are executed normally,
        # transfer tools return Command objects that exit the node.
        while hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    | Triage calling: {tc['name']}")

                # Check if it's a transfer tool
                transfer_tool_names = [t.name for t in transfer_tools]
                if tc["name"] in transfer_tool_names:
                    # Execute the transfer tool — it returns a Command
                    transfer_fn = next(t for t in transfer_tools if t.name == tc["name"])
                    cmd = transfer_fn.invoke(tc["args"])
                    print(f"    | -> Command(goto={cmd.goto})")
                    # Return the Command to let LangGraph handle routing
                    return cmd

            # Regular domain tool execution
            tool_node = ToolNode(triage_domain_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = triage_llm.invoke(messages, config=config)

        # If the LLM never called a transfer tool, go to report
        print("    | Triage did not call a transfer tool. Going to report.")
        return Command(
            goto="report",
            update={
                "messages": [response],
                "current_agent": "triage",
                "handoff_history": state["handoff_history"] + ["triage"],
                "handoff_depth": state["handoff_depth"] + 1,
            },
        )

    # ── Node: pharmacology ──────────────────────────────────────────────
    def pharmacology_node(state: CommandState) -> Command:
        """
        Pharmacology agent with domain tools and transfer_to_report.
        """
        print("\n    [STAGE 3.4] PHARMACOLOGY AGENT (with handoff tools)")
        print("    " + "-" * 50)

        handoff = state.get("handoff_context", {})
        print(f"    Received handoff: {json.dumps(handoff, indent=2, default=str)[:200]}")

        transfer_tools = create_transfer_tools(
            state["handoff_depth"], state["max_handoff_depth"]
        )
        # Pharmacology only gets transfer_to_report
        report_tool = [t for t in transfer_tools if t.name == "transfer_to_report"]
        all_tools = pharma_domain_tools + report_tool
        pharma_llm = llm.bind_tools(all_tools)

        findings = handoff.get("urgent_findings", [])
        reason = handoff.get("reason", "Direct pharmacology review")

        system_msg = SystemMessage(content=(
            "You are a clinical pharmacologist. Review medications for "
            "interactions and dosing safety. After completing your review, "
            "call transfer_to_report with a summary of your findings."
        ))

        user_msg = HumanMessage(content=f"""Handoff reason: {reason}

Findings to address:
{json.dumps(findings, indent=2)}

Medications:
{chr(10).join('  - ' + m for m in patient.current_medications)}

Labs: K+ = {patient.lab_results.get('K+')}, eGFR = {patient.lab_results.get('eGFR')}

Use your tools, then call transfer_to_report with your summary.""")

        config = build_callback_config(trace_name="handoff_cmd_pharma")
        messages = [system_msg, user_msg]
        response = pharma_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    | Pharmacology calling: {tc['name']}")

                if tc["name"] == "transfer_to_report":
                    transfer_fn = report_tool[0]
                    cmd = transfer_fn.invoke(tc["args"])
                    print(f"    | -> Command(goto={cmd.goto})")
                    return cmd

            tool_node = ToolNode(pharma_domain_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = pharma_llm.invoke(messages, config=config)

        # Fallback if no transfer called
        print("    | Pharmacology completed without transfer. Going to report.")
        return Command(
            goto="report",
            update={
                "messages": [response],
                "current_agent": "pharmacology",
                "handoff_history": state["handoff_history"] + ["pharmacology"],
                "handoff_depth": state["handoff_depth"] + 1,
            },
        )

    # ── Node: report ────────────────────────────────────────────────────
    def report_node(state: CommandState) -> dict:
        """Synthesise findings into a clinical summary."""
        print("\n    [STAGE 3.5] REPORT GENERATOR")
        print("    " + "-" * 50)

        handoff = state.get("handoff_context", {})
        summary = handoff.get("summary", "No summary provided.")

        agent_outputs = [
            msg.content for msg in state["messages"]
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls
        ]
        synthesis_input = "\n---\n".join(agent_outputs[-3:])

        report_llm = get_llm()
        config = build_callback_config(trace_name="handoff_cmd_report")
        response = report_llm.invoke(
            f"Synthesise into a clinical summary (max 150 words).\n\n"
            f"Handoff summary: {summary}\n\n"
            f"Agent outputs:\n{synthesis_input}",
            config=config,
        )

        print(f"\n    Report ({len(response.content)} chars):")
        for line in response.content.split("\n")[:8]:
            print(f"    | {line}")

        return {"final_report": response.content}

    # ── Wire the graph ──────────────────────────────────────────────────
    # With Command-based routing, we still register nodes.
    # But the edges FROM triage and pharmacology are driven by
    # the Command objects they return — not by add_edge().
    #
    # We only need add_edge() for START -> triage and report -> END,
    # because those are always fixed.

    workflow = StateGraph(CommandState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("pharmacology", pharmacology_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "triage")
    # triage -> (Command decides)
    # pharmacology -> (Command decides)
    workflow.add_edge("report", END)

    return workflow.compile(), patient


# ============================================================
# STAGE 3.7 — Execute
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  COMMAND-BASED HANDOFF")
    print("  Pattern: LLM decides routing via transfer tools + Command")
    print("=" * 70)

    print("""
    Graph topology:

        [START]
           |
           v
        [triage]  -- LLM calls transfer_to_pharmacology() -->
           |                                                   |
           |                                                   v
           |                                        [pharmacology]
           |                                                   |
           |         LLM calls transfer_to_report() <----------+
           |                    |
           v                    v
                          [report]
                              |
                              v
                            [END]

    Difference from conditional_routing.py:
      - No router function — the LLM calls a transfer tool instead.
      - Transfer tools return Command(goto=, update=).
      - LangGraph reads the Command and routes accordingly.
      - The LLM has autonomy to choose which specialist (or skip).
    """)

    graph, patient = build_command_pipeline()

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    K+={patient.lab_results['K+']} | eGFR={patient.lab_results['eGFR']}")
    print()

    initial_state: CommandState = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "handoff_context": {},
        "current_agent": "none",
        "handoff_history": [],
        "handoff_depth": 0,
        "max_handoff_depth": 4,
        "final_report": "",
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("  COMMAND HANDOFF COMPLETE")
    print("=" * 70)
    print(f"    Handoff chain  : {' -> '.join(result['handoff_history'])}")
    print(f"    Final depth    : {result['handoff_depth']}")

    print("""
    What happened:
      1. Triage agent had BOTH domain tools and transfer tools.
      2. After assessment, it called transfer_to_pharmacology().
      3. That tool returned Command(goto="pharmacology", update={...}).
      4. LangGraph routed to pharmacology without a conditional edge.
      5. Pharmacology called transfer_to_report() to finish.

    Key difference from earlier patterns:
      - linear_pipeline.py     : developer decides (add_edge)
      - conditional_routing.py : Python function decides (router)
      - command_handoff.py     : LLM decides (Command via tool)

    Next: supervisor.py — a coordinator LLM dispatches to workers.
    """)


if __name__ == "__main__":
    main()
