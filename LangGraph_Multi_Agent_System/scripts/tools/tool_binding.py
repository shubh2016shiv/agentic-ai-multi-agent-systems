#!/usr/bin/env python3
"""
============================================================
Tool Binding and Context Scoping
============================================================

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Demonstrate why each agent should receive only the tools
relevant to its role. bind_tools() returns a NEW LLM object
whose schema lists only the bound tools. Agents with fewer
tools make better tool selections.

This is "context scoping" at the tool level: narrowing the
LLM's action space to reduce confusion and improve accuracy.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    [Same LLM]  ── bind_tools([triage_tools])  ──>  [triage_llm]
                                                       (sees 2 tools)
    [Same LLM]  ── bind_tools([pharma_tools])  ──>  [pharma_llm]
                                                       (sees 3 tools)
    [Same LLM]  ── bind_tools([ALL tools])     ──>  [over_scoped_llm]
                                                       (sees 5+ tools)

    When you give an LLM fewer specific tools, it makes
    better tool selections. Over-scoping leads to:
      - Wrong tool calls (pharmacology tool on a triage question)
      - Wasted tokens (LLM processes irrelevant schemas)
      - Lower accuracy (more options = more confusion)

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. bind_tools() returns a NEW LLM instance (immutable pattern)
    2. Tool schema JSON — what the LLM actually sees
    3. Scoped agent vs over-scoped agent — accuracy comparison
    4. Per-agent tool sets for separation of concerns

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.tools.tool_binding
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import os
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.prebuilt import ToolNode

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage

# ── Project imports ─────────────────────────────────────────────────────────
from core.config import get_llm
from core.models import PatientCase
from tools import (
    analyze_symptoms,
    assess_patient_risk,
    check_drug_interactions,
    lookup_drug_info,
    calculate_dosage_adjustment,
    lookup_clinical_guideline,
)
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 6.1 — Define Tool Sets
# ============================================================
# Each agent gets a specific subset of the project's tools.
# The grouping reflects domain expertise.

TRIAGE_TOOLS = [analyze_symptoms, assess_patient_risk]
PHARMA_TOOLS = [check_drug_interactions, lookup_drug_info, calculate_dosage_adjustment]
GUIDELINE_TOOLS = [lookup_clinical_guideline]
ALL_TOOLS = TRIAGE_TOOLS + PHARMA_TOOLS + GUIDELINE_TOOLS


# ============================================================
# STAGE 6.2 — Show bind_tools() Creates a New Object
# ============================================================

def demonstrate_binding():
    """
    Show that bind_tools() returns a new LLM instance.

    The original LLM is unchanged. Each bound LLM has a
    different tool schema.
    """
    print("\n    [STAGE 6.2] bind_tools() returns a NEW LLM object")
    print("    " + "-" * 50)

    llm = get_llm()
    triage_llm = llm.bind_tools(TRIAGE_TOOLS)
    pharma_llm = llm.bind_tools(PHARMA_TOOLS)

    # They are different objects
    print(f"    llm is triage_llm? {llm is triage_llm}")       # False
    print(f"    llm is pharma_llm? {llm is pharma_llm}")       # False
    print(f"    triage_llm is pharma_llm? {triage_llm is pharma_llm}")  # False

    # The original is unmodified
    has_tools = hasattr(llm, "kwargs") and "tools" in getattr(llm, "kwargs", {})
    print(f"    Original LLM has tools bound? {has_tools}")  # False

    print("\n    Key takeaway: bind_tools() is IMMUTABLE.")
    print("    It does not mutate the original LLM.")
    print("    This lets you reuse the same base LLM across agents.")


# ============================================================
# STAGE 6.3 — Inspect Tool Schemas
# ============================================================

def show_tool_schemas():
    """
    Print the JSON schema that gets injected into the LLM prompt.

    This is what the LLM "sees" — tool names, descriptions,
    and parameter schemas. More tools = more schema text =
    more token consumption and potential confusion.
    """
    print("\n\n    [STAGE 6.3] Tool schemas — what the LLM sees")
    print("    " + "-" * 50)

    print("\n    TRIAGE TOOLS (2 tools):")
    for tool in TRIAGE_TOOLS:
        schema = tool.args_schema.model_json_schema() if hasattr(tool, "args_schema") else {}
        print(f"      {tool.name}:")
        print(f"        desc: {tool.description[:80]}...")
        params = list(schema.get("properties", {}).keys())
        print(f"        params: {params}")

    print("\n    PHARMA TOOLS (3 tools):")
    for tool in PHARMA_TOOLS:
        schema = tool.args_schema.model_json_schema() if hasattr(tool, "args_schema") else {}
        print(f"      {tool.name}:")
        print(f"        desc: {tool.description[:80]}...")
        params = list(schema.get("properties", {}).keys())
        print(f"        params: {params}")

    print(f"\n    ALL TOOLS ({len(ALL_TOOLS)} tools):")
    for tool in ALL_TOOLS:
        print(f"      - {tool.name}")

    print(f"\n    Schema count comparison:")
    print(f"      Triage agent sees {len(TRIAGE_TOOLS)} tool schemas")
    print(f"      Pharma agent sees {len(PHARMA_TOOLS)} tool schemas")
    print(f"      Over-scoped agent sees {len(ALL_TOOLS)} tool schemas")


# ============================================================
# STAGE 6.4 — Scoped vs Over-Scoped Comparison
# ============================================================

def run_comparison():
    """
    Run the same triage prompt through:
      1. A scoped agent (triage tools only)
      2. An over-scoped agent (all tools)

    Compare which tools each agent selects.
    """
    print("\n\n    [STAGE 6.4] Scoped vs over-scoped agent comparison")
    print("    " + "-" * 50)

    llm = get_llm()
    patient = PatientCase(
        patient_id="PT-TB-001",
        age=58, sex="M",
        chief_complaint="Persistent cough and dyspnea for 3 weeks",
        symptoms=["cough", "dyspnea", "wheezing", "fatigue"],
        medical_history=["COPD Stage II", "Former smoker (30 pack-years)"],
        current_medications=["Tiotropium 18mcg inhaler daily"],
        allergies=[],
        lab_results={"FEV1": "58% predicted", "SpO2": "93%"},
        vitals={"BP": "138/85", "HR": "92", "SpO2": "93%"},
    )

    triage_prompt = f"""Evaluate this patient for triage:
Patient: {patient.age}y {patient.sex}
Complaint: {patient.chief_complaint}
Symptoms: {', '.join(patient.symptoms)}
Labs: {json.dumps(patient.lab_results)}

Use available tools to assess this patient."""

    system = SystemMessage(content="You are a triage specialist. Evaluate the patient.")

    # ── Run 1: Scoped (triage tools only) ─────────────────────────────
    print("\n    RUN 1: SCOPED AGENT (2 triage tools)")
    scoped_llm = llm.bind_tools(TRIAGE_TOOLS)
    config = build_callback_config(trace_name="tool_binding_scoped")
    response = scoped_llm.invoke(
        [system, HumanMessage(content=triage_prompt)],
        config=config,
    )

    scoped_calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            scoped_calls.append(tc["name"])
            print(f"      Called: {tc['name']}")
    else:
        print("      (No tool calls — LLM responded directly)")

    # ── Run 2: Over-scoped (all tools) ────────────────────────────────
    print("\n    RUN 2: OVER-SCOPED AGENT (all tools)")
    over_llm = llm.bind_tools(ALL_TOOLS)
    config = build_callback_config(trace_name="tool_binding_overscoped")
    response = over_llm.invoke(
        [system, HumanMessage(content=triage_prompt)],
        config=config,
    )

    over_calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            over_calls.append(tc["name"])
            print(f"      Called: {tc['name']}")
    else:
        print("      (No tool calls — LLM responded directly)")

    # ── Compare ───────────────────────────────────────────────────────
    triage_names = {t.name for t in TRIAGE_TOOLS}
    scoped_correct = all(c in triage_names for c in scoped_calls)
    over_correct = all(c in triage_names for c in over_calls)

    print("\n    COMPARISON:")
    print(f"      Scoped agent tools called  : {scoped_calls}")
    print(f"      Over-scoped tools called   : {over_calls}")
    print(f"      Scoped stayed in domain?   : {scoped_correct}")
    print(f"      Over-scoped in domain?     : {over_correct}")

    if not over_correct:
        wrong = [c for c in over_calls if c not in triage_names]
        print(f"      Over-scoped called wrong tools: {wrong}")
        print("      -> This is why context scoping matters.")


# ============================================================
# STAGE 6.5 — Summary
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  TOOL BINDING AND CONTEXT SCOPING")
    print("  Pattern: restrict each agent's tool set via bind_tools()")
    print("=" * 70)

    demonstrate_binding()
    show_tool_schemas()
    run_comparison()

    print("\n\n" + "=" * 70)
    print("  TOOL BINDING COMPLETE")
    print("=" * 70)
    print("""
    What you saw:
      1. bind_tools() is immutable — returns a new LLM, doesn't modify original.
      2. Each tool set has its own JSON schema injected into the prompt.
      3. Scoped agents call the right tools for their domain.
      4. Over-scoped agents may call irrelevant tools.

    Why context scoping matters:
      - Accuracy: fewer tools = less confusion for the LLM.
      - Cost: smaller tool schemas = fewer input tokens.
      - Accountability: each agent is responsible for its domain.
      - Security: agents can't accidentally call tools outside their scope.

    Next: toolnode_patterns.py — two ways to execute tools in LangGraph.
    """)


if __name__ == "__main__":
    main()
