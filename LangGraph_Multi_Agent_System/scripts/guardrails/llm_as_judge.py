#!/usr/bin/env python3
"""
============================================================
LLM-as-Judge
============================================================
Pattern E: Model-based guardrails — use a second LLM call
to evaluate the agent's response for safety, relevance,
and completeness.
Prerequisite: layered_validation.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
The previous scripts used deterministic guardrails (regex,
keyword matching) to validate inputs and outputs. These are
fast and predictable, but they cannot assess nuance:

    - Is the response RELEVANT to the specific patient case?
    - Is the clinical reasoning COMPLETE?
    - Is the recommendation SAFE in context?

An LLM judge evaluates these semantic qualities by reading
the original query and the agent's response, then returning
a structured verdict.

Trade-off:
    Deterministic guardrails: fast, cheap, no false positives
                              on exact patterns, but blind to
                              semantic issues.
    LLM-as-judge:            slower, costs tokens, can assess
                              meaning, but may itself hallucinate.

Production systems use BOTH: deterministic checks first
(fast, cheap), LLM judge second (on responses that passed
the deterministic check).

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [agent]            <-- LLM call 1: generate clinical assessment
       |
       v
    [judge]            <-- LLM call 2: evaluate the response
       |
    route_after_judge()
       |
    +--+---------+----------+
    |             |          |
    | "approve"   | "revise" | "reject"
    v             v          v
    [deliver]   [revise]   [reject]
    |             |          |
    v             v          v
    [END]        [END]     [END]

    DECISION TABLE:
        verdict="approve"  -> deliver as-is
        verdict="revise"   -> append judge's suggested fix
        verdict="reject"   -> replace with safe fallback

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. LLM-as-judge with structured output (Pydantic model)
    2. Two-LLM pattern: agent + judge
    3. Semantic evaluation vs deterministic checks
    4. When to combine both approaches

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.guardrails.llm_as_judge
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import json
from typing import Literal, TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage

# ── Project imports ─────────────────────────────────────────────────────────
from core.config import get_llm
from core.models import PatientCase
from observability.callbacks import build_callback_config

# CONNECTION: JudgeVerdict schema and evaluate_with_judge() live in the root
# guardrails module. JudgeVerdict is a Pydantic model with fields: safety,
# relevance, completeness, verdict, reasoning, suggested_fix.
# evaluate_with_judge() wraps the second LLM call + structured output parsing.
# See guardrails/llm_judge_guardrails.py for the concept explanation and the
# JUDGE_SYSTEM_PROMPT that drives the evaluation rubric.
from guardrails.llm_judge_guardrails import (
    JudgeVerdict,
    evaluate_with_judge,
    default_approve_verdict,
)


# ============================================================
# STAGE 5.1 — Judge Verdict Schema
# ============================================================
# JudgeVerdict is imported from guardrails.llm_judge_guardrails (root module).
#
# CONCEPT: JudgeVerdict is a Pydantic model used with
# llm.with_structured_output(JudgeVerdict). This forces the judge LLM to
# return a validated object with fields:
#   safety       — "safe" | "unsafe" | "borderline"
#   relevance    — "relevant" | "partially_relevant" | "irrelevant"
#   completeness — "complete" | "partial" | "incomplete"
#   verdict      — "approve" | "revise" | "reject"
#   reasoning    — str (one-paragraph explanation)
#   suggested_fix — str (empty unless verdict="revise")
#
# Using structured output eliminates the need to parse free-text responses
# from the judge — the Pydantic model validates format automatically.
# See guardrails/llm_judge_guardrails.py for the full schema and docstrings.


# ============================================================
# STAGE 5.2 — State Definition
# ============================================================

class JudgeState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    user_query: str            # Original query
    agent_response: str        # Written by: agent_node
    judge_verdict: dict        # Written by: judge_node (serialised JudgeVerdict)
    final_output: str          # Written by: deliver/revise/reject
    status: str                # Written by: terminal nodes


# ============================================================
# STAGE 5.3 — Node Definitions
# ============================================================

def agent_node(state: JudgeState) -> dict:
    """
    LLM Call 1: Generate clinical assessment.

    The agent produces a response to the patient query.
    This response will then be evaluated by the judge.
    """
    llm = get_llm()
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a clinical triage specialist. "
        "Provide a concise clinical assessment for the patient below."
    ))
    prompt = HumanMessage(content=f"""Patient: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
History: {', '.join(patient.get('medical_history', []))}
Medications: {', '.join(patient.get('current_medications', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
Vitals: {json.dumps(patient.get('vitals', {}))}

Query: {state['user_query']}""")

    config = build_callback_config(trace_name="llm_judge_agent")
    response = llm.invoke([system, prompt], config=config)

    print(f"    | [Agent] Response: {len(response.content)} chars")

    return {
        "messages": [response],
        "agent_response": response.content,
    }


def judge_node(state: JudgeState) -> dict:
    """
    LLM Call 2: Evaluate the agent's response using the root module judge.

    CONNECTION: evaluate_with_judge() from guardrails.llm_judge_guardrails
    encapsulates the second LLM call. It receives the patient case, user query,
    and agent response, then returns a structured JudgeVerdict.

    This node is a thin wrapper: it calls the root module function and writes
    the serialised verdict to graph state. The graph topology (routing from
    judge → deliver/revise/reject) remains in THIS script — that is the pattern.

    Two-LLM pattern in action:
        LLM Call 1: agent_node → generates clinical assessment
        LLM Call 2: judge_node → evaluate_with_judge() → structured verdict
    """
    llm = get_llm()
    config = build_callback_config(trace_name="llm_judge_verdict")

    # CONNECTION: evaluate_with_judge() wraps the judge LLM call with
    # structured output parsing. default_approve_verdict() provides a
    # fail-open fallback if the judge LLM errors (fail-open is safe here
    # because deterministic guardrails are the first line of defense).
    try:
        verdict = evaluate_with_judge(
            llm=llm,
            patient_case=state["patient_case"],
            user_query=state["user_query"],
            agent_response=state["agent_response"],
        )

        print(f"    | [Judge] Safety: {verdict.safety}")
        print(f"    | [Judge] Relevance: {verdict.relevance}")
        print(f"    | [Judge] Completeness: {verdict.completeness}")
        print(f"    | [Judge] Verdict: {verdict.verdict}")
        print(f"    | [Judge] Reasoning: {verdict.reasoning[:100]}...")

        return {"judge_verdict": verdict.model_dump()}

    except RuntimeError as e:
        # fail-open: deterministic guardrails (Layer 1) already ran
        print(f"    | [Judge] Error: {e}. Using fail-open default (approve).")
        fallback = default_approve_verdict(reason=str(e))
        return {"judge_verdict": fallback.model_dump()}


def route_after_judge(state: JudgeState) -> Literal["deliver", "revise", "reject"]:
    """
    Route based on the judge's verdict.

    "approve" -> deliver as-is
    "revise"  -> apply suggested fix
    "reject"  -> replace with safe fallback
    """
    verdict = state["judge_verdict"].get("verdict", "approve")
    return verdict if verdict in ("approve", "revise", "reject") else "deliver"


def deliver_node(state: JudgeState) -> dict:
    """Deliver response — judge approved."""
    return {
        "final_output": state["agent_response"],
        "status": "approved",
    }


def revise_node(state: JudgeState) -> dict:
    """
    Apply the judge's suggested fix.

    In a more advanced implementation, this node could call
    the LLM again with the judge's feedback to generate a
    revised response. Here we append the fix as a note.
    """
    fix = state["judge_verdict"].get("suggested_fix", "")
    reasoning = state["judge_verdict"].get("reasoning", "")

    revised = (
        f"{state['agent_response']}\n\n"
        f"--- Revision Note ---\n"
        f"Reviewer comment: {reasoning}\n"
    )
    if fix:
        revised += f"Suggested correction: {fix}\n"

    return {
        "final_output": revised,
        "status": "revised",
    }


def reject_node(state: JudgeState) -> dict:
    """Replace response — judge found it fundamentally unsafe."""
    reasoning = state["judge_verdict"].get("reasoning", "")
    return {
        "final_output": (
            "This response was rejected by clinical review.\n"
            f"Reason: {reasoning}\n"
            "Please consult a qualified healthcare provider directly."
        ),
        "status": "rejected",
    }


# ============================================================
# STAGE 5.4 — Graph Construction
# ============================================================

def build_judge_graph():
    """
    Build and compile the LLM-as-judge graph.

    Graph: START → agent → judge → (3-way) → deliver/revise/reject → END
    """
    workflow = StateGraph(JudgeState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("deliver", deliver_node)
    workflow.add_node("revise", revise_node)
    workflow.add_node("reject", reject_node)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "judge")
    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {"approve": "deliver", "deliver": "deliver", "revise": "revise", "reject": "reject"},
    )
    workflow.add_edge("deliver", END)
    workflow.add_edge("revise", END)
    workflow.add_edge("reject", END)

    return workflow.compile()


# ============================================================
# STAGE 5.5 — Test Cases
# ============================================================

def make_state(query: str, patient: PatientCase) -> JudgeState:
    return {
        "messages": [],
        "patient_case": patient.model_dump(),
        "user_query": query,
        "agent_response": "",
        "judge_verdict": {},
        "final_output": "",
        "status": "pending",
    }


def main() -> None:
    print("\n" + "=" * 70)
    print("  LLM-AS-JUDGE")
    print("  Pattern: model-based guardrails")
    print("=" * 70)

    print("""
    Two LLM calls per request:

        [agent]  -> LLM Call 1: generate clinical assessment
                       |
        [judge]  -> LLM Call 2: evaluate safety, relevance, completeness
                       |
                  route_after_judge()
                       |
              approve / revise / reject

    Deterministic vs model-based guardrails:
        Deterministic (regex) : fast, cheap, catches exact patterns
        LLM-as-judge          : slower, costs tokens, catches semantic issues

    Production recommendation: use BOTH.
        Layer 1: Deterministic checks (input_validation, output_validation)
        Layer 2: LLM judge (only on responses that pass deterministic checks)
    """)

    graph = build_judge_graph()

    # ── Test 1: Standard clinical query ───────────────────────────────
    print("=" * 70)
    print("  TEST 1: Standard COPD assessment")
    print("=" * 70)

    copd_patient = PatientCase(
        patient_id="PT-JG-001",
        age=58, sex="M",
        chief_complaint="Persistent cough and dyspnea for 3 weeks",
        symptoms=["cough", "dyspnea", "wheezing"],
        medical_history=["COPD Stage II", "Former smoker"],
        current_medications=["Tiotropium 18mcg"],
        allergies=[],
        lab_results={"FEV1": "58% predicted", "SpO2": "93%"},
        vitals={"BP": "138/85", "HR": "92"},
    )

    r1 = graph.invoke(make_state(
        "Assess this COPD patient and recommend next steps.",
        copd_patient,
    ))
    print(f"\n    STATUS: {r1['status'].upper()}")
    print(f"    Judge verdict: {r1['judge_verdict'].get('verdict', 'unknown')}")

    # ── Test 2: Complex polypharmacy case ─────────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 2: Complex polypharmacy — drug interaction risk")
    print("=" * 70)

    poly_patient = PatientCase(
        patient_id="PT-JG-002",
        age=71, sex="F",
        chief_complaint="Dizziness with elevated potassium",
        symptoms=["dizziness", "fatigue", "ankle edema"],
        medical_history=["CKD Stage 3a", "Hypertension", "CHF"],
        current_medications=[
            "Lisinopril 20mg", "Spironolactone 25mg",
            "Furosemide 40mg", "Metoprolol 50mg",
        ],
        allergies=["Sulfa drugs"],
        lab_results={"K+": "5.4 mEq/L", "eGFR": "42 mL/min", "BNP": "450 pg/mL"},
        vitals={"BP": "105/65", "HR": "58"},
    )

    r2 = graph.invoke(make_state(
        "Assess this patient. The combination of Lisinopril and Spironolactone "
        "with declining renal function is concerning. What are the risks?",
        poly_patient,
    ))
    print(f"\n    STATUS: {r2['status'].upper()}")
    print(f"    Judge verdict: {r2['judge_verdict'].get('verdict', 'unknown')}")

    # ── Test 3: Vague query, minimal data ───────────────────────────
    print("\n\n" + "=" * 70)
    print("  TEST 3: Vague query with minimal patient data")
    print("=" * 70)

    vague_patient = PatientCase(
        patient_id="PT-JG-003",
        age=30, sex="M",
        chief_complaint="Feeling unwell",
        symptoms=["fatigue"],
        medical_history=[],
        current_medications=[],
        allergies=[],
        lab_results={},
        vitals={},
    )

    r3 = graph.invoke(make_state(
        "What's wrong with this patient?",
        vague_patient,
    ))
    print(f"\n    STATUS: {r3['status'].upper()}")
    print(f"    Judge verdict: {r3['judge_verdict'].get('verdict', 'unknown')}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  LLM-AS-JUDGE COMPLETE")
    print("=" * 70)
    print(f"""
    Results:
      Test 1 (clear COPD)       : {r1['status'].upper()} ({r1['judge_verdict'].get('verdict', 'unknown')})
      Test 2 (polypharmacy)     : {r2['status'].upper()} ({r2['judge_verdict'].get('verdict', 'unknown')})
      Test 3 (vague/minimal)    : {r3['status'].upper()} ({r3['judge_verdict'].get('verdict', 'unknown')})

    Judge evaluation dimensions:
      Safety       — is the advice medically safe?
      Relevance    — does it address THIS patient's case?
      Completeness — are key clinical considerations covered?

    Deterministic vs model-based:
      Deterministic: catches "stop all medications immediately" (exact match)
      LLM judge: catches "the recommendation doesn't account for the
                  patient's CKD when suggesting dosage" (semantic)

    Cost considerations:
      Each request costs 2 LLM calls (agent + judge).
      In production, run the judge only on responses that:
        - Pass deterministic checks first
        - Involve high-risk clinical decisions
        - Fall below a confidence threshold

    This completes the guardrail pattern series.
    """)


if __name__ == "__main__":
    main()
