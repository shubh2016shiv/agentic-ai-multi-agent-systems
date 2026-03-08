#!/usr/bin/env python3
"""
============================================================
Adversarial Debate
============================================================
Pattern 4: Two agents argue opposing viewpoints on a clinical
decision. A judge agent synthesizes and rules.

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
In complex clinical decisions (surgery vs. medical management,
aggressive vs. conservative treatment), anchoring bias is
a real risk. The debate pattern forces consideration of BOTH
perspectives:

    1. Pro agent argues FOR a treatment
    2. Con agent argues AGAINST it
    3. Both agents rebut each other's arguments
    4. Judge agent weighs evidence and rules

This mirrors medical grand rounds and tumor board discussions.

This is ideal when:
    - Treatment decisions have legitimate opposing views
    - Anchoring bias must be actively countered
    - You need documented rationale for both sides

When NOT to use:
    - Straightforward decisions (use pipeline)
    - When you need consensus, not adversarial testing (use voting)

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [opening_arguments]    <-- pro + con agents argue independently
       |
       v
    [rebuttals]            <-- each agent rebuts the other
       |
       v
    [judge_ruling]         <-- judge weighs all arguments
       |
       v
    [END]

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()    opening_args     rebuttals       judge
      |            |               |              |
      |-- invoke ->|               |              |
      |            |-- Pro: for    |              |
      |            |-- Con: against|              |
      |            |---- state --->|              |
      |            |               |-- Pro rebuts |
      |            |               |-- Con rebuts |
      |            |               |--- state --->|
      |            |               |              |-- weigh args
      |            |               |              |-- ruling
      |<-- verdict + rationale ----|--------------|

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.MAS_architectures.adversarial_debate
============================================================
"""

# -- Standard library --------------------------------------------------------
import sys
import json
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
# STAGE 4.1 -- State Definition
# ============================================================

class DebateState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    debate_question: str
    pro_opening: str
    con_opening: str
    pro_rebuttal: str
    con_rebuttal: str
    judge_verdict: str


# ============================================================
# STAGE 4.2 -- Node Definitions
# ============================================================

def opening_arguments_node(state: DebateState) -> dict:
    """
    Round 1: Both sides present opening arguments independently.

    Pro agent argues FOR the proposed treatment.
    Con agent argues AGAINST it.
    Neither sees the other's argument (prevents anchoring).
    """
    llm = get_llm()
    patient = state["patient_case"]
    question = state["debate_question"]

    patient_summary = (
        f"{patient.get('age')}y {patient.get('sex')}, "
        f"{patient.get('chief_complaint')}, "
        f"History: {', '.join(patient.get('medical_history', []))}, "
        f"Labs: {json.dumps(patient.get('lab_results', {}))}"
    )

    # Pro argument
    pro_prompt = f"""You are an interventional specialist ARGUING FOR aggressive treatment.

Debate Question: {question}
Patient: {patient_summary}

Make your strongest case for intervention. Cite specific clinical evidence, 
guidelines, and risk/benefit analysis. 120 words max."""

    config_pro = build_callback_config(trace_name="debate_pro_opening", tags=["debate", "pro"])
    pro_response = llm.invoke(pro_prompt, config=config_pro)
    print(f"    | [Pro] Opening: {pro_response.content[:120]}...")

    # Con argument
    con_prompt = f"""You are a conservative medicine specialist ARGUING AGAINST aggressive treatment.

Debate Question: {question}
Patient: {patient_summary}

Make your strongest case for conservative/medical management. Cite specific 
clinical evidence, risks of intervention, and alternative approaches. 120 words max."""

    config_con = build_callback_config(trace_name="debate_con_opening", tags=["debate", "con"])
    con_response = llm.invoke(con_prompt, config=config_con)
    print(f"    | [Con] Opening: {con_response.content[:120]}...")

    return {
        "pro_opening": pro_response.content,
        "con_opening": con_response.content,
    }


def rebuttals_node(state: DebateState) -> dict:
    """
    Round 2: Each side rebuts the other's opening argument.

    Now each agent CAN see the opposing argument and must
    directly address its strongest points.
    """
    llm = get_llm()

    # Pro rebuts Con
    pro_rebuttal_prompt = f"""Your opponent argued AGAINST intervention:

"{state['con_opening']}"

Provide a focused rebuttal addressing their specific points. 
Where are they wrong or overlooking evidence? 80 words max."""

    config_pro = build_callback_config(trace_name="debate_pro_rebuttal", tags=["debate", "pro"])
    pro_rebuttal = llm.invoke(pro_rebuttal_prompt, config=config_pro)
    print(f"    | [Pro] Rebuttal: {pro_rebuttal.content[:100]}...")

    # Con rebuts Pro
    con_rebuttal_prompt = f"""Your opponent argued FOR intervention:

"{state['pro_opening']}"

Provide a focused rebuttal addressing their specific points.
Where are they wrong or overstating benefits? 80 words max."""

    config_con = build_callback_config(trace_name="debate_con_rebuttal", tags=["debate", "con"])
    con_rebuttal = llm.invoke(con_rebuttal_prompt, config=config_con)
    print(f"    | [Con] Rebuttal: {con_rebuttal.content[:100]}...")

    return {
        "pro_rebuttal": pro_rebuttal.content,
        "con_rebuttal": con_rebuttal.content,
    }


def judge_ruling_node(state: DebateState) -> dict:
    """
    Judge -- weighs all arguments and issues a ruling.

    The judge sees:
        1. The debate question
        2. Pro opening + rebuttal
        3. Con opening + rebuttal
        4. The patient data

    The judge must:
        1. Identify the strongest argument on each side
        2. State the tipping factor
        3. Issue a clear ruling
    """
    llm = get_llm()
    patient = state["patient_case"]

    judge_prompt = f"""You are an impartial clinical judge reviewing this medical debate.

DEBATE QUESTION: {state['debate_question']}

PATIENT: {patient.get('age')}y {patient.get('sex')}, {patient.get('chief_complaint')}
Labs: {json.dumps(patient.get('lab_results', {}))}

PRO OPENING (for intervention):
{state['pro_opening']}

PRO REBUTTAL:
{state['pro_rebuttal']}

CON OPENING (against intervention):
{state['con_opening']}

CON REBUTTAL:
{state['con_rebuttal']}

Issue your ruling:
1. STRONGEST PRO ARGUMENT: (one sentence)
2. STRONGEST CON ARGUMENT: (one sentence)
3. TIPPING FACTOR: What single factor tips the decision?
4. RULING: [FOR INTERVENTION] or [FOR CONSERVATIVE MANAGEMENT]
5. RATIONALE: (2-3 sentences)

Keep under 150 words."""

    config = build_callback_config(trace_name="debate_judge_ruling", tags=["debate", "judge"])
    response = llm.invoke(judge_prompt, config=config)
    print(f"    | [Judge] Verdict: {response.content[:120]}...")

    return {"judge_verdict": response.content}


# ============================================================
# STAGE 4.3 -- Graph Construction
# ============================================================

def build_debate_graph():
    """
    Build the adversarial debate graph.

    Structure: opening -> rebuttals -> judge
    All edges are fixed (no conditional routing).
    The adversarial nature comes from prompt design, not graph topology.
    """
    workflow = StateGraph(DebateState)

    workflow.add_node("opening_arguments", opening_arguments_node)
    workflow.add_node("rebuttals", rebuttals_node)
    workflow.add_node("judge_ruling", judge_ruling_node)

    workflow.add_edge(START, "opening_arguments")
    workflow.add_edge("opening_arguments", "rebuttals")
    workflow.add_edge("rebuttals", "judge_ruling")
    workflow.add_edge("judge_ruling", END)

    return workflow.compile()


# ============================================================
# STAGE 4.4 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  ADVERSARIAL DEBATE")
    print("  Pattern: opposing arguments + judge ruling")
    print("=" * 70)

    print("""
    Architecture:

        [Pro Agent]                 [Con Agent]
            |                           |
            +-----> Round 1: Opening <--+
            |                           |
            +-----> Round 2: Rebuttal <-+
                         |
                    [Judge Agent]
                         |
                      Verdict

    Two rounds of argument, then an impartial ruling.
    """)

    patient = PatientCase(
        patient_id="PT-ARCH-004",
        age=76, sex="M",
        chief_complaint="Increasingly exercise intolerant with syncopal episode",
        symptoms=["dyspnea", "fatigue", "syncope", "chest pain"],
        medical_history=["Moderate Aortic Stenosis", "CKD Stage 3b", "Hypertension", "Diabetes"],
        current_medications=["Lisinopril 10mg", "Metformin 500mg BID", "Aspirin 81mg"],
        allergies=["Contrast dye"],
        lab_results={"eGFR": "35 mL/min", "Troponin": "0.02 ng/mL", "BNP": "450 pg/mL", "HbA1c": "7.5%"},
        vitals={"BP": "130/75", "HR": "68", "SpO2": "95%"},
    )

    debate_question = (
        "Should this 76-year-old with moderate-severe aortic stenosis, CKD 3b, "
        "and a syncopal episode undergo transcatheter aortic valve replacement (TAVR) "
        "or be managed conservatively with medications?"
    )

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "debate_question": debate_question,
        "pro_opening": "",
        "con_opening": "",
        "pro_rebuttal": "",
        "con_rebuttal": "",
        "judge_verdict": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Question: {debate_question[:100]}...")
    print()
    print("    " + "-" * 60)

    graph = build_debate_graph()
    result = graph.invoke(initial_state)

    # -- Display full debate -----------------------------------------------
    print("\n    " + "=" * 60)
    print("    DEBATE TRANSCRIPT")
    print("    " + "-" * 60)

    print("    PRO (For Intervention):")
    for line in result["pro_opening"][:300].split("\n"):
        if line.strip():
            print(f"      {line}")
    print()
    print("    CON (Against Intervention):")
    for line in result["con_opening"][:300].split("\n"):
        if line.strip():
            print(f"      {line}")

    print("\n    " + "-" * 60)
    print("    REBUTTALS:")
    print(f"    Pro: {result['pro_rebuttal'][:150]}...")
    print(f"    Con: {result['con_rebuttal'][:150]}...")

    # -- Display verdict ---------------------------------------------------
    print("\n    " + "=" * 60)
    print("    JUDGE VERDICT")
    print("    " + "-" * 60)
    for line in result["judge_verdict"].split("\n"):
        print(f"    | {line}")

    # -- Summary -----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  ADVERSARIAL DEBATE SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. Debate = adversarial prompt design, not graph complexity
      2. Pro and Con agents argue independently, then rebut
      3. Judge agent sees ALL arguments before ruling
      4. Two-round structure: opening + rebuttal prevents shallow analysis
      5. Documents both sides of the decision (audit trail)

    When to use:
      - Treatment dilemmas with legitimate opposing views
      - When you need documented rationale for both sides
      - Tumor board / grand rounds simulation

    When NOT to use:
      - Clear-cut decisions (unnecessary overhead)
      - When consensus is needed (use voting instead)

    Next: hierarchical_delegation.py
    """)


if __name__ == "__main__":
    main()
