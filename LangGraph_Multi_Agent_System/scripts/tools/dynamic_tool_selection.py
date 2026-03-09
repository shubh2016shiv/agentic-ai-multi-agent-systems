#!/usr/bin/env python3
"""
============================================================
Dynamic Tool Selection
============================================================
Prerequisite: structured_output.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Select tools at runtime based on patient data instead of
hardcoding them at graph-build time. A "tool selector" node
inspects the patient's conditions and picks the relevant
tool set before the agent runs.

Static binding (bind_tools at build time) works when you know
which agent handles which case. Dynamic selection works when
the same agent might need different tools depending on input.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    [START]
       |
       v
    [tool_selector]     <-- inspects patient data, picks tool set
       |
       | writes: selected_tools to state
       v
    [agent]             <-- bind_tools() with dynamically-selected tools
       |
       v
     [END]

    The tool_selector is a regular Python node (no LLM call).
    The agent uses whichever tools were selected.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Tool registry — mapping conditions to tool groups
    2. Runtime tool selection node
    3. Dynamic bind_tools() based on state
    4. When to use static vs dynamic binding

------------------------------------------------------------
WHEN TO USE
------------------------------------------------------------
    Use dynamic_tool_selection when the same agent may need
    different tools depending on the runtime input (e.g. patient
    conditions determine which tool domain is relevant).

    When NOT to use:
    - If the tool set is always the same per agent (use tool_binding.py)
    - If you always know which agent handles which case at build time

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.tools.dynamic_tool_selection
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import os
import json
from typing import TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── Project imports ─────────────────────────────────────────────────────────
# CONNECTION: core/ root module — get_llm() centralises LLM config.
# PatientCase is the canonical domain model; its conditions field drives
# the dynamic tool registry lookup in this pattern.
from core.config import get_llm
from core.models import PatientCase
# CONNECTION: tools/ root module — the full tool library is registered in
# TOOL_REGISTRY and selected subsets are bound at runtime. This script demos
# WHEN to bind which tools, not what the tools do. See tools/ for implementations.
from tools import (
    analyze_symptoms,
    assess_patient_risk,
    check_drug_interactions,
    lookup_drug_info,
    calculate_dosage_adjustment,
    lookup_clinical_guideline,
)
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse trace_name and tags to every LLM call automatically.
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 9.1 — Tool Registry
# ============================================================
# Maps condition keywords to tool groups. When a patient's
# conditions match a keyword, those tools become available.

TOOL_REGISTRY = {
    "renal": {
        "description": "Renal dosing and drug interaction tools",
        "keywords": ["ckd", "renal", "egfr", "creatinine", "kidney"],
        "tools": [check_drug_interactions, calculate_dosage_adjustment, lookup_drug_info],
    },
    "cardiac": {
        "description": "Cardiac assessment tools",
        "keywords": ["heart", "cardiac", "hypertension", "edema", "bnp", "chf"],
        "tools": [analyze_symptoms, assess_patient_risk],
    },
    "respiratory": {
        "description": "Respiratory assessment tools",
        "keywords": ["copd", "asthma", "dyspnea", "cough", "fev1", "spo2"],
        "tools": [analyze_symptoms, assess_patient_risk],
    },
    "guideline": {
        "description": "Clinical guideline lookup",
        "keywords": ["guideline", "protocol", "standard"],
        "tools": [lookup_clinical_guideline],
    },
    "polypharmacy": {
        "description": "Multi-drug interaction tools",
        "keywords": ["polypharmacy", "multiple medications"],
        "tools": [check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment],
    },
}

# Baseline tools always included
BASELINE_TOOLS = [analyze_symptoms, assess_patient_risk]


# ============================================================
# STAGE 9.2 — State Definition
# ============================================================

class DynamicToolState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    selected_tool_names: list[str]   # names of selected tools (for logging)
    matched_categories: list[str]    # which registry categories matched
    agent_response: str


# ============================================================
# STAGE 9.3 — Build Graph
# ============================================================

def build_dynamic_pipeline():
    """Build a graph with runtime tool selection."""
    llm = get_llm()

    # ── Node: tool_selector ─────────────────────────────────────────────
    def tool_selector_node(state: DynamicToolState) -> dict:
        """
        Inspect the patient data and select tools from the registry.

        This is a pure Python node — no LLM call. It scans the
        patient's history, symptoms, labs, and medications for
        keywords that match registry categories.
        """
        print("\n    [STAGE 9.2] TOOL SELECTOR")
        print("    " + "-" * 50)

        patient = state["patient_case"]

        # Build a searchable text from patient data
        search_text = " ".join([
            patient.get("chief_complaint", ""),
            " ".join(patient.get("symptoms", [])),
            " ".join(patient.get("medical_history", [])),
            " ".join(patient.get("current_medications", [])),
            json.dumps(patient.get("lab_results", {})),
        ]).lower()

        # Scan registry for matches (use tool names for set — StructuredTool is unhashable)
        matched_categories = []
        selected_tool_names: set[str] = set()

        for category, config in TOOL_REGISTRY.items():
            for keyword in config["keywords"]:
                if keyword in search_text:
                    matched_categories.append(category)
                    for tool in config["tools"]:
                        selected_tool_names.add(tool.name)
                    print(f"    | Matched '{keyword}' -> category: {category}")
                    break

        # Always include baseline tools
        for tool in BASELINE_TOOLS:
            selected_tool_names.add(tool.name)

        # Resolve names back to tool list for logging
        all_tools_by_name = {t.name: t for t in
                             BASELINE_TOOLS + [check_drug_interactions, lookup_drug_info,
                                               calculate_dosage_adjustment, lookup_clinical_guideline]}
        tool_list = [all_tools_by_name[n] for n in selected_tool_names]
        tool_names = list(selected_tool_names)

        print(f"\n    Matched categories : {matched_categories}")
        print(f"    Selected tools ({len(tool_list)}):")
        for t in tool_list:
            print(f"      - {t.name}")

        return {
            "selected_tool_names": tool_names,
            "matched_categories": matched_categories,
        }

    # ── Node: agent ─────────────────────────────────────────────────────
    def agent_node(state: DynamicToolState) -> dict:
        """
        Agent that uses dynamically-selected tools.

        Reads selected_tool_names from state, resolves them
        to actual tool functions, and calls bind_tools() at
        execution time (not build time).
        """
        print("\n    [STAGE 9.3] AGENT (dynamic tools)")
        print("    " + "-" * 50)

        # Resolve tool names to functions
        all_available = {t.name: t for t in
                         BASELINE_TOOLS + [check_drug_interactions, lookup_drug_info,
                                           calculate_dosage_adjustment, lookup_clinical_guideline]}

        selected_names = state.get("selected_tool_names", [])
        tools = [all_available[name] for name in selected_names if name in all_available]

        print(f"    Tools bound: {[t.name for t in tools]}")

        # Bind at runtime
        agent_llm = llm.bind_tools(tools)

        patient = state["patient_case"]
        system_msg = SystemMessage(content=(
            "You are a clinical specialist. Evaluate the patient using "
            "your tools. Provide a concise assessment."
        ))
        user_msg = HumanMessage(content=f"""Patient: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
Medications: {', '.join(patient.get('current_medications', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
History: {', '.join(patient.get('medical_history', []))}

Use your available tools to assess this patient.""")

        config = build_callback_config(trace_name="dynamic_tool_agent")
        messages = [system_msg, user_msg]
        response = agent_llm.invoke(messages, config=config)

        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | Calling {len(response.tool_calls)} tool(s):")
            for tc in response.tool_calls:
                print(f"      -> {tc['name']}")

            tool_node = ToolNode(tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = agent_llm.invoke(messages, config=config)

        print(f"\n    Assessment: {len(response.content)} chars")

        return {
            "messages": [response],
            "agent_response": response.content,
        }

    # ── Wire ────────────────────────────────────────────────────────────
    workflow = StateGraph(DynamicToolState)
    workflow.add_node("tool_selector", tool_selector_node)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "tool_selector")
    workflow.add_edge("tool_selector", "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


# ============================================================
# STAGE 9.4 — Test Cases
# ============================================================

def make_state(patient: PatientCase) -> DynamicToolState:
    return {
        "messages": [],
        "patient_case": patient.model_dump(),
        "selected_tool_names": [],
        "matched_categories": [],
        "agent_response": "",
    }


def run_renal_patient(graph) -> None:
    """Patient with CKD — triggers renal tools."""
    print("\n" + "=" * 70)
    print("  TEST 1: RENAL PATIENT (CKD, high K+)")
    print("  Expected tools: renal + cardiac (baseline + drug interaction)")
    print("=" * 70)

    patient = PatientCase(
        patient_id="PT-DTS-RENAL",
        age=71, sex="F",
        chief_complaint="Dizziness with elevated K+",
        symptoms=["dizziness", "fatigue", "edema"],
        medical_history=["CKD Stage 3a", "Hypertension"],
        current_medications=["Lisinopril 20mg", "Spironolactone 25mg"],
        allergies=[],
        lab_results={"K+": "5.4 mEq/L", "eGFR": "42 mL/min"},
        vitals={"BP": "105/65"},
    )

    result = graph.invoke(make_state(patient))
    print(f"\n    Categories: {result['matched_categories']}")
    print(f"    Tools used: {result['selected_tool_names']}")


def run_respiratory_patient(graph) -> None:
    """Patient with COPD — triggers respiratory tools."""
    print("\n\n" + "=" * 70)
    print("  TEST 2: RESPIRATORY PATIENT (COPD, dyspnea)")
    print("  Expected tools: respiratory baseline only")
    print("=" * 70)

    patient = PatientCase(
        patient_id="PT-DTS-RESP",
        age=58, sex="M",
        chief_complaint="Worsening COPD exacerbation",
        symptoms=["dyspnea", "cough", "wheezing"],
        medical_history=["COPD Stage II"],
        current_medications=["Tiotropium 18mcg"],
        allergies=[],
        lab_results={"FEV1": "45% predicted", "SpO2": "90%"},
        vitals={"BP": "145/85", "HR": "100"},
    )

    result = graph.invoke(make_state(patient))
    print(f"\n    Categories: {result['matched_categories']}")
    print(f"    Tools used: {result['selected_tool_names']}")


# ============================================================
# STAGE 9.5 — Summary
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  DYNAMIC TOOL SELECTION")
    print("  Pattern: choose tools at runtime based on patient data")
    print("=" * 70)

    print("""
    Graph:

        [START] -> [tool_selector] -> [agent] -> [END]
                        |                |
                  (no LLM call)    (bind_tools with
                  scans patient     selected tools)
                  data for
                  keywords)

    Tool Registry:
      renal       -> drug interaction, dosage adjustment, drug info
      cardiac     -> analyze symptoms, assess risk
      respiratory -> analyze symptoms, assess risk
      guideline   -> clinical guideline lookup
    """)

    graph = build_dynamic_pipeline()

    run_renal_patient(graph)
    run_respiratory_patient(graph)

    print("\n\n" + "=" * 70)
    print("  DYNAMIC TOOL SELECTION COMPLETE")
    print("=" * 70)
    print("""
    What you saw:
      Test 1: CKD patient -> renal + cardiac tools selected
      Test 2: COPD patient -> respiratory tools selected

    Static vs dynamic binding:
      Static  : bind_tools() at graph build time. Fixed tools.
                Use when agents have permanent domain expertise.
      Dynamic : bind_tools() at runtime from state. Variable tools.
                Use when the same node handles different specialties
                depending on input data.

    Production considerations:
      - Keep the tool registry small and well-tested.
      - Log which tools were selected (audit trail).
      - Consider cost: more tools = more schema tokens.

    Next: tool_error_handling.py — handling tool failures.
    """)


if __name__ == "__main__":
    main()
