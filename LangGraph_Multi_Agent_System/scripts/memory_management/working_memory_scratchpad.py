#!/usr/bin/env python3
"""
============================================================
Working Memory Scratchpad
============================================================
Pattern 1: Inter-agent scratchpad using WorkingMemory.
Multiple agents share a key-value store through LangGraph
state, reading upstream findings and writing their own.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Understand how WorkingMemory serves as a shared scratchpad
in a multi-agent pipeline:

    Agent A writes findings → Agent B reads them → Agent C
    reads everything and compiles a report.

No LLM calls in this script. The focus is purely on memory
mechanics: serialisation, deserialisation, shared vs private
namespaces, and the append pattern for audit trails.

------------------------------------------------------------
MEMORY MODEL
------------------------------------------------------------

    WorkingMemory = in-memory dict that flows through state

    ┌─────────────────────────────────────────────────────┐
    │  state["working_memory"] = {                        │
    │      "triage_findings": "...",        # shared      │
    │      "risk_level": "high",            # shared      │
    │      "medications_reviewed": [...],   # shared      │
    │      "reasoning_trace": [...],        # append-only │
    │      "_scratch": {                    # private     │
    │          "triage": {"temp_calc": 42}, # agent-local │
    │          "pharma": {"lookup": "..."}  # agent-local │
    │      }                                              │
    │  }                                                  │
    └─────────────────────────────────────────────────────┘

    Shared namespace: any agent can read/write
    Private scratch:  agent-local temporary data
    Append-only:      reasoning_trace accumulates entries

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [triage_agent]       <-- WRITES: triage_findings, risk_level
       |                     APPENDS: reasoning_trace
       v
    [pharmacist_agent]   <-- READS: triage_findings, risk_level
       |                     WRITES: medications_reviewed
       v                     APPENDS: reasoning_trace
    [report_agent]       <-- READS: ALL keys via to_context_string()
       |                     WRITES: report_complete
       v
    [END]

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()           triage_agent      pharmacist_agent   report_agent
      |                   |                  |                 |
      |--- invoke() ----->|                  |                 |
      |                   |-- load_working_memory(state)       |
      |                   |-- set(triage_findings)             |
      |                   |-- set(risk_level)                  |
      |                   |-- append_to(reasoning_trace)       |
      |                   |-- save_working_memory(working_mem) |
      |                   |------ state ------>|               |
      |                   |                    |-- load(state) |
      |                   |                    |-- get(triage) |
      |                   |                    |-- set(warns)  |
      |                   |                    |-- append(tr)  |
      |                   |                    |-- save(mem)   |
      |                   |                    |----- state -->|
      |                   |                    |               |-- load(state)
      |                   |                    |               |-- to_context_string()
      |                   |                    |               |-- set(complete)
      |<------------ final state ----------------------------- |
      |                   |                  |                 |

------------------------------------------------------------
KEY CONCEPTS
------------------------------------------------------------
    1. Serialise WorkingMemory as dict in state (JSON-safe)
    2. load_working_memory() / save_working_memory() pattern
    3. Shared namespace for inter-agent data
    4. Private scratch space for agent-local temporaries
    5. append_to() for audit trail / reasoning trace
    6. to_context_string() for injecting memory into prompts

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.memory_management.working_memory_scratchpad
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import json
from typing import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END

# CONNECTION: WorkingMemory lives in the root memory module (Tier 2 memory).
# It provides get(), set(), append_to(), get_scratch(), to_context_string().
# This script demonstrates how to carry a WorkingMemory instance across nodes
# using LangGraph state — the pattern, not the WorkingMemory implementation.
# See memory/working_memory.py for the "WHERE THIS FITS" concept explanation.
from memory.working_memory import WorkingMemory


# ============================================================
# STAGE 1.1 — State Definition
# ============================================================

class ScratchpadState(TypedDict):
    patient_case: dict              # Set at invocation
    working_memory: dict            # Serialised WorkingMemory (shared)
    triage_output: str              # Written by: triage_agent
    pharmacist_output: str          # Written by: pharmacist_agent
    final_report: str               # Written by: report_agent


# ============================================================
# STAGE 1.2 — Serialisation Helpers
# ============================================================
# These two functions are the canonical pattern for WorkingMemory
# in LangGraph state. Every script in this series uses them.

def load_working_memory(state: dict) -> WorkingMemory:
    """Deserialise WorkingMemory from state dict."""
    return WorkingMemory(initial_data=state.get("working_memory", {}))


def save_working_memory(working_mem: WorkingMemory) -> dict:
    """Serialise WorkingMemory back to a plain dict for state."""
    return working_mem.get_all()


# ============================================================
# STAGE 1.3 — Node Definitions
# ============================================================

def triage_agent(state: ScratchpadState) -> dict:
    """
    Triage agent — simulated (no LLM).

    WRITES to shared WorkingMemory:
        triage_findings: summary of triage assessment
        risk_level: "high" / "moderate" / "low"
        patient_symptoms: list of symptoms

    WRITES to private scratch:
        triage._raw_vitals: temporary data not needed downstream

    APPENDS to reasoning_trace:
        Audit entry for this step
    """
    patient = state["patient_case"]
    working_mem = load_working_memory(state)

    # ── Simulated triage logic ───────────────────────────────────────
    symptoms = patient.get("symptoms", [])
    vitals = patient.get("vitals", {})

    # Determine risk level from vitals
    spo2_str = vitals.get("SpO2", "99%").replace("%", "")
    try:
        spo2 = int(spo2_str)
    except ValueError:
        spo2 = 99

    risk_level = "high" if spo2 < 94 else "moderate" if spo2 < 97 else "low"

    findings = (
        f"Patient {patient.get('age')}y {patient.get('sex')} presents with "
        f"{patient.get('chief_complaint')}. "
        f"Symptoms: {', '.join(symptoms)}. "
        f"SpO2: {spo2}%. Risk: {risk_level}."
    )

    # ── Write to SHARED WorkingMemory ────────────────────────────────
    working_mem.set("triage_findings", findings)
    working_mem.set("risk_level", risk_level)
    working_mem.set("patient_symptoms", symptoms)
    working_mem.set("patient_medications", patient.get("current_medications", []))

    # ── Write to PRIVATE scratch (agent-local) ───────────────────────
    working_mem.set_scratch("triage", "raw_vitals", vitals)
    working_mem.set_scratch("triage", "spo2_numeric", spo2)

    # ── Append to reasoning trace (audit trail) ──────────────────────
    working_mem.append_to("reasoning_trace", f"Triage: {risk_level} risk, SpO2={spo2}%")

    print(f"    | [Triage] Findings: {findings[:80]}...")
    print(f"    | [Triage] Risk level: {risk_level}")
    print(f"    | [Triage] Working Memory keys (shared): {working_mem.keys()}")
    print(f"    | [Triage] Scratch keys: {list(working_mem.get_scratch('triage').keys())}")

    return {
        "working_memory": save_working_memory(working_mem),
        "triage_output": findings,
    }


def pharmacist_agent(state: ScratchpadState) -> dict:
    """
    Pharmacist agent — simulated (no LLM).

    READS from shared WorkingMemory:
        triage_findings: to understand the clinical picture
        risk_level: to adjust medication review urgency
        patient_medications: to check interactions

    WRITES to shared WorkingMemory:
        medications_reviewed: list of reviewed medications
        interaction_warnings: any flagged drug interactions

    APPENDS to reasoning_trace:
        Audit entry for this step
    """
    working_mem = load_working_memory(state)

    # ── READ from upstream agent's shared data ───────────────────────
    triage_findings = working_mem.get("triage_findings", "No triage data available")
    risk_level = working_mem.get("risk_level", "unknown")
    medications = working_mem.get("patient_medications", [])

    print(f"    | [Pharmacist] Read triage findings: {triage_findings[:60]}...")
    print(f"    | [Pharmacist] Read risk level: {risk_level}")
    print(f"    | [Pharmacist] Read medications: {medications}")

    # ── Simulated drug interaction check ─────────────────────────────
    interaction_warnings = []
    medication_names_lower = [m.lower() for m in medications]

    if any("lisinopril" in name for name in medication_names_lower) and any("spironolactone" in name for name in medication_names_lower):
        interaction_warnings.append(
            "WARNING: Lisinopril + Spironolactone may cause hyperkalemia. "
            "Monitor K+ levels closely."
        )
    if any("metformin" in name for name in medication_names_lower):
        interaction_warnings.append(
            "NOTE: Metformin requires eGFR monitoring. Hold if eGFR < 30 mL/min."
        )
    if not interaction_warnings:
        interaction_warnings.append("No significant drug interactions identified.")

    review_summary = (
        f"Reviewed {len(medications)} medications. "
        f"Found {len(interaction_warnings)} note(s). "
        f"Risk context: {risk_level}."
    )

    # ── Write to SHARED WorkingMemory ────────────────────────────────
    working_mem.set("medications_reviewed", medications)
    working_mem.set("interaction_warnings", interaction_warnings)

    # ── Private scratch (agent-local) ────────────────────────────────
    working_mem.set_scratch("pharma", "raw_med_count", len(medications))

    # ── Append to reasoning trace ────────────────────────────────────
    working_mem.append_to("reasoning_trace", f"Pharma: {len(interaction_warnings)} warnings for {len(medications)} meds")

    print(f"    | [Pharmacist] Warnings: {interaction_warnings}")
    print(f"    | [Pharmacist] Working Memory keys: {working_mem.keys()}")

    return {
        "working_memory": save_working_memory(working_mem),
        "pharmacist_output": review_summary,
    }


def report_agent(state: ScratchpadState) -> dict:
    """
    Report agent — reads FULL WorkingMemory via to_context_string().

    This agent demonstrates the consumer pattern: it reads everything
    that upstream agents wrote, without needing to know which agent
    wrote what. The to_context_string() method produces a formatted
    summary suitable for LLM prompt injection.
    """
    working_mem = load_working_memory(state)

    # ── Read the FULL context ────────────────────────────────────────
    full_context = working_mem.to_context_string(max_length=2000)
    print(f"    | [Report] Full Working Memory context ({len(full_context)} chars):")
    for line in full_context.split("\n")[:15]:
        if line.strip():
            print(f"    |   {line}")

    # ── Read specific keys for structured report ─────────────────────
    triage_findings = working_mem.get("triage_findings", "N/A")
    risk_level = working_mem.get("risk_level", "N/A")
    interaction_warnings = working_mem.get("interaction_warnings", [])
    reasoning_trace = working_mem.get("reasoning_trace", [])

    report = (
        f"CLINICAL SUMMARY\n"
        f"{'=' * 40}\n"
        f"Triage: {triage_findings}\n\n"
        f"Risk Level: {risk_level.upper()}\n\n"
        f"Medication Review:\n"
    )
    for warning in interaction_warnings:
        report += f"  - {warning}\n"
    report += f"\nReasoning Trace ({len(reasoning_trace)} steps):\n"
    for step_index, entry in enumerate(reasoning_trace):
        report += f"  [{step_index+1}] {entry}\n"

    # ── Mark report complete ─────────────────────────────────────────
    working_mem.append_to("reasoning_trace", "Report: Clinical summary generated")
    working_mem.set("report_complete", True)

    print(f"    | [Report] Report length: {len(report)} chars")

    return {
        "working_memory": save_working_memory(working_mem),
        "final_report": report,
    }


# ============================================================
# STAGE 1.4 — Graph Construction
# ============================================================

def build_scratchpad_graph():
    """
    Build the scratchpad demonstration graph.

    Three agents in sequence, sharing WorkingMemory through state.
    """
    workflow = StateGraph(ScratchpadState)

    workflow.add_node("triage_agent", triage_agent)
    workflow.add_node("pharmacist_agent", pharmacist_agent)
    workflow.add_node("report_agent", report_agent)

    workflow.add_edge(START, "triage_agent")
    workflow.add_edge("triage_agent", "pharmacist_agent")
    workflow.add_edge("pharmacist_agent", "report_agent")
    workflow.add_edge("report_agent", END)

    return workflow.compile()


# ============================================================
# STAGE 1.5 — Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  WORKING MEMORY SCRATCHPAD")
    print("  Pattern: inter-agent shared memory via WorkingMemory")
    print("=" * 70)

    print("""
    Three agents share WorkingMemory through LangGraph state:

      [triage_agent]      WRITES: triage_findings, risk_level
             |            APPENDS: reasoning_trace
             v
      [pharmacist_agent]  READS: triage_findings, medications
             |            WRITES: interaction_warnings
             v            APPENDS: reasoning_trace
      [report_agent]      READS: everything (to_context_string)
             |            WRITES: report_complete
             v
           [END]

    Serialisation pattern:
      Each node does:
        working_mem = load_working_memory(state)       # deserialise
        working_mem.set("key", value)                   # read/write
        return {"working_memory": save_working_memory(working_mem)}  # serialise back

    This pattern ensures WorkingMemory survives LangGraph
    state merging and checkpointer serialisation.
    """)

    # ── Test patient ──────────────────────────────────────────────────
    patient = {
        "patient_id": "PT-WM-001",
        "age": 71, "sex": "F",
        "chief_complaint": "Dizziness with elevated potassium",
        "symptoms": ["dizziness", "fatigue", "ankle edema"],
        "medical_history": ["CKD Stage 3a", "Hypertension", "CHF"],
        "current_medications": [
            "Lisinopril 20mg daily",
            "Spironolactone 25mg daily",
            "Furosemide 40mg daily",
            "Metformin 500mg twice daily",
        ],
        "allergies": ["Sulfa drugs"],
        "lab_results": {"K+": "5.4 mEq/L", "eGFR": "42 mL/min", "BNP": "450 pg/mL"},
        "vitals": {"BP": "105/65", "HR": "58", "SpO2": "93%"},
    }

    initial_state = {
        "patient_case": patient,
        "working_memory": {},
        "triage_output": "",
        "pharmacist_output": "",
        "final_report": "",
    }

    graph = build_scratchpad_graph()
    print("    Graph compiled.\n")
    print("    " + "-" * 60)

    result = graph.invoke(initial_state)

    # ── Display results ───────────────────────────────────────────────
    print("\n    " + "=" * 60)
    print("    PIPELINE RESULTS")
    print("    " + "=" * 60)

    print(f"\n    FINAL REPORT:")
    print(f"    {'─' * 50}")
    for line in result["final_report"].split("\n"):
        print(f"    | {line}")

    # ── Show final WorkingMemory state ────────────────────────────────
    working_memory_final = result.get("working_memory", {})
    print(f"\n    WORKING MEMORY — FINAL STATE:")
    print(f"    Keys: {list(working_memory_final.keys())}")
    if "reasoning_trace" in working_memory_final:
        print(f"    Reasoning trace ({len(working_memory_final['reasoning_trace'])} entries):")
        for step_index, entry in enumerate(working_memory_final["reasoning_trace"]):
            print(f"      [{step_index}] {entry}")

    print("\n\n" + "=" * 70)
    print("  WORKING MEMORY SCRATCHPAD COMPLETE")
    print("=" * 70)
    print("""
    What you saw:
      1. triage_agent WROTE findings to shared namespace
      2. pharmacist_agent READ triage findings and WROTE warnings
      3. report_agent READ everything via to_context_string()
      4. reasoning_trace accumulated entries from all agents

    Key patterns:
      - load_working_memory() / save_working_memory() for serialisation
      - working_mem.set() / working_mem.get() for shared data
      - working_mem.set_scratch() for agent-private temporaries
      - working_mem.append_to() for audit trails
      - working_mem.to_context_string() for LLM prompt injection

    Next: checkpoint_persistence.py — cross-invocation state.
    """)


if __name__ == "__main__":
    main()
