#!/usr/bin/env python3
"""
============================================================
Reflection and Self-Critique
============================================================
Pattern 7: Generate-critique-revise loop with conditional
iteration and max-iteration guard.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
In clinical decision support, a single-pass LLM response
may contain errors, omissions, or safety concerns. The
reflection pattern adds a quality improvement loop:

    1. GENERATOR produces an initial clinical recommendation
    2. CRITIC evaluates it for errors, omissions, safety issues
    3. If the critique is severe, the GENERATOR revises
    4. The loop repeats until quality is acceptable or max iterations reached

This is the simplest form of self-improvement in LLM systems.

This is ideal when:
    - Output quality must exceed single-pass capability
    - Safety-critical outputs need automated review before delivery
    - You want iterative refinement without human intervention

When NOT to use:
    - When single-pass quality is sufficient (added latency)
    - When human review is mandatory anyway (use HITL instead)

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [generator]  <---------+
       |                   |
       v                   |
    [critic]               |
       |                   |
    route_after_critique() |
       |                   |
    +--+---+               |
    |      |               |
    | "revise" --> [generator]   (loop back)
    |      |
    | "accept" --> [END]

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()     generator         critic           route
      |            |                |                |
      |-- invoke ->|                |                |
      |            |-- LLM(initial)|                |
      |            |---- state ---->|                |
      |            |                |-- LLM(critique)|
      |            |                |---- state ---->|
      |            |                |                |-- "revise"
      |            |<---- loop back --------|        |
      |            |-- LLM(revised)|                |
      |            |---- state ---->|                |
      |            |                |-- LLM(critique)|
      |            |                |---- state ---->|
      |            |                |                |-- "accept"
      |<-- final --|                |                |

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.MAS_architectures.reflection_self_critique
============================================================
"""

# -- Standard library --------------------------------------------------------
import sys
import json
import re
from typing import TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -- LangGraph ---------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# -- Project imports ----------------------------------------------------------
from core.config import get_llm
from core.models import PatientCase
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 7.1 -- State Definition
# ============================================================

class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    current_recommendation: str       # latest generator output
    critique: str                     # latest critic output
    critique_severity: str            # "minor" or "major"
    revision_history: list[str]       # all versions for audit trail
    iteration: int
    max_iterations: int
    is_accepted: bool


# ============================================================
# STAGE 7.2 -- Node Definitions
# ============================================================

def generator_node(state: ReflectionState) -> dict:
    """
    Generator agent -- produces or revises the clinical recommendation.

    On first call: generates from scratch using patient data.
    On subsequent calls: revises based on the critic's feedback.
    """
    llm = get_llm()
    patient = state["patient_case"]
    iteration = state.get("iteration", 0)
    prior_critique = state.get("critique", "")
    prior_recommendation = state.get("current_recommendation", "")
    revision_history = list(state.get("revision_history", []))

    if iteration == 0:
        # First pass: generate from scratch
        prompt = f"""Provide a treatment plan for this patient:

Patient: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
History: {', '.join(patient.get('medical_history', []))}
Medications: {', '.join(patient.get('current_medications', []))}
Allergies: {', '.join(patient.get('allergies', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
Vitals: {json.dumps(patient.get('vitals', {}))}

Include: 1) Immediate actions 2) Medication changes 3) Monitoring plan 4) Follow-up
Keep under 150 words."""

        label = "initial"
    else:
        # Revision: address the critic's concerns
        prompt = f"""Your previous recommendation was criticized:

PREVIOUS RECOMMENDATION:
{prior_recommendation}

CRITIQUE:
{prior_critique}

Revise the recommendation to address ALL critique points.
Maintain what was already correct. Fix what was wrong or missing.
Keep under 150 words."""

        label = f"revision_{iteration}"

    config = build_callback_config(
        trace_name=f"reflection_generator_{label}",
        tags=["reflection", "generator"],
    )
    response = llm.invoke(prompt, config=config)

    revision_history.append(response.content)
    print(f"    | [Generator] Iteration {iteration}: {response.content[:100]}...")

    return {
        "current_recommendation": response.content,
        "revision_history": revision_history,
        "iteration": iteration + 1,
    }


def critic_node(state: ReflectionState) -> dict:
    """
    Critic agent -- evaluates the generator's recommendation.

    Checks for:
        1. Clinical errors or inaccuracies
        2. Missing critical elements (allergies, interactions)
        3. Safety concerns (dose too high, contraindications)
        4. Completeness (all required sections present)

    Returns a severity: "major" (needs revision) or "minor" (acceptable).
    """
    llm = get_llm()
    patient = state["patient_case"]
    recommendation = state.get("current_recommendation", "")
    iteration = state.get("iteration", 0)

    prompt = f"""Critically evaluate this clinical recommendation:

RECOMMENDATION:
{recommendation}

PATIENT CONTEXT:
Age: {patient.get('age')}y {patient.get('sex')}
Allergies: {', '.join(patient.get('allergies', []))}
Current Medications: {', '.join(patient.get('current_medications', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}

Check for:
1. CLINICAL ERRORS: Are any facts wrong?
2. MISSING ELEMENTS: Does it address allergies, interactions, monitoring?
3. SAFETY CONCERNS: Are doses appropriate? Any contraindications missed?
4. COMPLETENESS: Are all required sections present?

End your critique with a severity assessment:
- "SEVERITY: MAJOR" if there are clinical errors or safety concerns
- "SEVERITY: MINOR" if only style or minor completeness issues remain

Keep under 120 words."""

    config = build_callback_config(
        trace_name=f"reflection_critic_{iteration}",
        tags=["reflection", "critic"],
    )
    response = llm.invoke(prompt, config=config)

    # Parse severity
    critique_text = response.content
    severity = "minor"  # default
    if "SEVERITY: MAJOR" in critique_text.upper() or "MAJOR" in critique_text.upper().split("SEVERITY")[-1] if "SEVERITY" in critique_text.upper() else False:
        severity = "major"

    print(f"    | [Critic] Severity: {severity.upper()} | {critique_text[:80]}...")

    return {
        "critique": critique_text,
        "critique_severity": severity,
    }


def route_after_critique(state: ReflectionState) -> str:
    """
    Routing logic: revise or accept based on critique severity
    and iteration count.

    Accept if:
        - Critique severity is "minor" (quality acceptable)
        - Max iterations reached (prevent infinite loops)

    Revise if:
        - Critique severity is "major" AND iterations remain
    """
    severity = state.get("critique_severity", "minor")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if severity == "major" and iteration < max_iterations:
        print(f"    | [Router] Major critique at iteration {iteration}, looping back to generator")
        return "revise"
    else:
        reason = "minor critique (acceptable)" if severity == "minor" else f"max iterations ({max_iterations}) reached"
        print(f"    | [Router] Accepting: {reason}")
        return "accept"


# ============================================================
# STAGE 7.3 -- Graph Construction
# ============================================================

def build_reflection_graph():
    """
    Build the generate-critique-revise loop.

    Key patterns:
        - Conditional edge from critic: "revise" loops back, "accept" ends
        - Max iteration guard prevents infinite loops
        - Revision history provides full audit trail
    """
    workflow = StateGraph(ReflectionState)

    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)

    workflow.add_edge(START, "generator")
    workflow.add_edge("generator", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_after_critique,
        {
            "revise": "generator",    # loop back
            "accept": END,            # finish
        },
    )

    return workflow.compile()


# ============================================================
# STAGE 7.4 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  REFLECTION AND SELF-CRITIQUE")
    print("  Pattern: generate -> critique -> revise loop")
    print("=" * 70)

    print("""
    Reflection loop:

        [Generator] --> [Critic] --+--> "accept" --> END
                                   |
                                   +--> "revise" --> [Generator] (loop)

    Max iterations prevent infinite loops.
    Each revision is stored in revision_history for audit.
    """)

    patient = PatientCase(
        patient_id="PT-ARCH-007",
        age=76, sex="M",
        chief_complaint="Increasingly exercise intolerant with syncopal episode",
        symptoms=["dyspnea", "fatigue", "syncope", "chest pain"],
        medical_history=["Moderate Aortic Stenosis", "CKD Stage 3b", "Hypertension", "Diabetes"],
        current_medications=["Lisinopril 10mg", "Metformin 500mg BID", "Aspirin 81mg"],
        allergies=["Contrast dye"],
        lab_results={"eGFR": "35 mL/min", "Troponin": "0.02 ng/mL", "BNP": "450 pg/mL", "HbA1c": "7.5%"},
        vitals={"BP": "130/75", "HR": "68", "SpO2": "95%"},
    )

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "current_recommendation": "",
        "critique": "",
        "critique_severity": "",
        "revision_history": [],
        "iteration": 0,
        "max_iterations": 3,
        "is_accepted": False,
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print(f"    Max iterations: 3")
    print(f"    Key risk: CKD 3b + contrast dye allergy")
    print()
    print("    " + "-" * 60)

    graph = build_reflection_graph()
    result = graph.invoke(initial_state)

    # -- Display revision history ------------------------------------------
    print("\n    " + "=" * 60)
    print("    REVISION HISTORY")
    print("    " + "-" * 60)
    revision_history = result.get("revision_history", [])
    for version_index, version_text in enumerate(revision_history):
        version_label = "INITIAL" if version_index == 0 else f"REVISION {version_index}"
        print(f"\n    [{version_label}]:")
        for line in version_text[:300].split("\n"):
            if line.strip():
                print(f"      {line}")

    # -- Display final critique -------------------------------------------
    print(f"\n    " + "=" * 60)
    print("    FINAL CRITIQUE")
    print("    " + "-" * 60)
    print(f"    Severity: {result.get('critique_severity', '?').upper()}")
    for line in result.get("critique", "")[:300].split("\n"):
        if line.strip():
            print(f"    | {line}")

    # -- Display final accepted recommendation -----------------------------
    print(f"\n    " + "=" * 60)
    print("    ACCEPTED RECOMMENDATION")
    print("    " + "-" * 60)
    for line in result.get("current_recommendation", "").split("\n"):
        print(f"    | {line}")

    # -- Stats -------------------------------------------------------------
    print(f"\n    Iterations: {result.get('iteration', 0)}")
    print(f"    Revisions: {len(revision_history) - 1}")
    print(f"    Final severity: {result.get('critique_severity', '?').upper()}")

    # -- Summary -----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  REFLECTION AND SELF-CRITIQUE SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Generator produces or revises recommendations
      2. Critic evaluates for errors, omissions, safety
      3. Conditional edge loops back on "major" critique
      4. Max iteration guard prevents infinite loops
      5. Revision history provides full audit trail

    When to use:
      - Safety-critical outputs needing automated review
      - When single-pass quality is insufficient
      - Iterative refinement without human intervention

    When NOT to use:
      - When human review is always required (use HITL)
      - When latency is critical (each loop = added LLM call)

    This completes the MAS Architecture Patterns series:
      1. supervisor_orchestration    -- dynamic routing
      2. sequential_pipeline         -- fixed-order processing
      3. parallel_voting             -- independent + consensus
      4. adversarial_debate          -- opposing views + judge
      5. hierarchical_delegation     -- multi-level management
      6. map_reduce_fanout           -- split + process + aggregate
      7. reflection_self_critique    -- generate + critique + revise
    """)


if __name__ == "__main__":
    main()
