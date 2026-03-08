#!/usr/bin/env python3
"""
============================================================
Structured Output
============================================================
Prerequisite: toolnode_patterns.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Get deterministic, validated JSON from LLM agents instead of
free-form text. Two patterns:

    Pattern 1: with_structured_output()
        Forces the LLM to return a Pydantic model.
        The LLM's JSON output is validated by Pydantic.

    Pattern 2: Response-format tool
        Define a "submit_result" tool whose args ARE the
        desired schema. The LLM calls this tool, and you
        extract the structured data from the tool call args.

Both produce a Pydantic-validated object. Pattern 1 is cleaner.
Pattern 2 is useful when you also need the LLM to call domain
tools before producing structured output.

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. with_structured_output() — cleanest approach
    2. Response-format tool pattern
    3. Pydantic model as output schema
    4. Validation: what happens with malformed output

------------------------------------------------------------
WHEN TO USE
------------------------------------------------------------
    Use Pattern 1 (with_structured_output) when you need clean
    validated JSON from the LLM with no other tool calls.

    Use Pattern 2 (response-format tool) when the LLM must also
    call domain tools before producing its structured output.

    When NOT to use:
    - If free-form text output is acceptable (no need for Pydantic)

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.tools.structured_output
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import os
import json
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.prebuilt import ToolNode

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# ── Project imports ─────────────────────────────────────────────────────────
# CONNECTION: core/ root module — get_llm() centralises LLM config.
# PatientCase is the canonical domain model used in test scenarios.
from core.config import get_llm
from core.models import PatientCase
# CONNECTION: tools/ root module — analyze_symptoms and assess_patient_risk are
# component-layer tools used in Pattern 2 (response-format tool).
# This script demos HOW to force structured output from the LLM.
from tools import analyze_symptoms, assess_patient_risk
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse trace_name and tags to every LLM call automatically.
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 8.1 — Define Output Schema
# ============================================================

class TriageResult(BaseModel):
    """
    Structured triage assessment output.

    This Pydantic model defines the EXACT shape of the
    output we want from the LLM. Fields have types,
    constraints, and descriptions.
    """
    urgency: Literal["emergent", "urgent", "semi-urgent", "non-urgent"] = Field(
        description="Triage urgency level"
    )
    risk_score: float = Field(
        ge=0.0, le=1.0,
        description="Risk score from 0.0 (lowest) to 1.0 (highest)"
    )
    primary_concern: str = Field(
        description="Single most important clinical finding"
    )
    findings: list[str] = Field(
        description="List of 2-5 key clinical findings"
    )
    recommended_actions: list[str] = Field(
        description="List of 2-4 recommended next steps"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Agent's confidence in this assessment"
    )


# ── Shared patient ──────────────────────────────────────────────────────────
PATIENT = PatientCase(
    patient_id="PT-SO-001",
    age=71, sex="F",
    chief_complaint="Dizziness and fatigue with elevated K+",
    symptoms=["dizziness", "fatigue", "ankle edema"],
    medical_history=["Hypertension", "CKD Stage 3a"],
    current_medications=["Lisinopril 20mg", "Spironolactone 25mg"],
    allergies=["Sulfa drugs"],
    lab_results={"K+": "5.4 mEq/L", "eGFR": "42 mL/min"},
    vitals={"BP": "105/65", "HR": "88"},
)

TRIAGE_PROMPT = f"""Evaluate this patient:
Patient: {PATIENT.age}y {PATIENT.sex}
Complaint: {PATIENT.chief_complaint}
Symptoms: {', '.join(PATIENT.symptoms)}
Medications: {', '.join(PATIENT.current_medications)}
Labs: K+={PATIENT.lab_results.get('K+')}, eGFR={PATIENT.lab_results.get('eGFR')}
Vitals: BP={PATIENT.vitals.get('BP')}, HR={PATIENT.vitals.get('HR')}

Provide a clinical triage assessment."""


# ============================================================
# STAGE 8.2 — Pattern 1: with_structured_output()
# ============================================================

def run_pattern_1():
    """
    Use with_structured_output() to force the LLM to return
    a TriageResult Pydantic model.

    The LLM's output is automatically parsed and validated
    by Pydantic. If the LLM produces invalid JSON, Pydantic
    raises a ValidationError.
    """
    print("\n    [STAGE 8.2] Pattern 1: with_structured_output()")
    print("    " + "-" * 50)

    llm = get_llm()
    structured_llm = llm.with_structured_output(TriageResult)

    config = build_callback_config(trace_name="structured_output_p1")

    try:
        result = structured_llm.invoke(
            [
                SystemMessage(content="You are a clinical triage specialist."),
                HumanMessage(content=TRIAGE_PROMPT),
            ],
            config=config,
        )

        print(f"    Type: {type(result).__name__}")
        print(f"    Urgency          : {result.urgency}")
        print(f"    Risk score       : {result.risk_score:.2f}")
        print(f"    Primary concern  : {result.primary_concern}")
        print(f"    Findings ({len(result.findings)}):")
        for f in result.findings:
            print(f"      - {f}")
        print(f"    Actions ({len(result.recommended_actions)}):")
        for a in result.recommended_actions:
            print(f"      - {a}")
        print(f"    Confidence       : {result.confidence:.0%}")

        return result

    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        return None


# ============================================================
# STAGE 8.3 — Pattern 2: Response-Format Tool
# ============================================================

def run_pattern_2():
    """
    Define a "submit" tool whose args ARE the output schema.
    The LLM calls this tool, and we extract the args as
    structured data.

    This pattern lets the LLM use domain tools first, then
    submit its findings in a structured format.
    """
    print("\n\n    [STAGE 8.3] Pattern 2: response-format tool")
    print("    " + "-" * 50)

    @tool
    def submit_triage_result(
        urgency: str,
        risk_score: float,
        primary_concern: str,
        findings: list[str],
        recommended_actions: list[str],
        confidence: float,
    ) -> str:
        """
        Submit the final triage assessment. Call this AFTER
        completing your analysis with domain tools.

        Args:
            urgency: One of: emergent, urgent, semi-urgent, non-urgent
            risk_score: 0.0 (lowest risk) to 1.0 (highest risk)
            primary_concern: The single most important finding
            findings: List of 2-5 key clinical findings
            recommended_actions: List of 2-4 recommended next steps
            confidence: Your confidence in this assessment (0.0 to 1.0)
        """
        return "Triage result submitted."

    llm = get_llm()
    domain_tools = [analyze_symptoms, assess_patient_risk]
    all_tools = domain_tools + [submit_triage_result]
    bound_llm = llm.bind_tools(all_tools)

    config = build_callback_config(trace_name="structured_output_p2")
    messages = [
        SystemMessage(content=(
            "You are a clinical triage specialist. First use your domain "
            "tools to assess the patient, then call submit_triage_result "
            "with your structured findings."
        )),
        HumanMessage(content=TRIAGE_PROMPT),
    ]

    response = bound_llm.invoke(messages, config=config)

    # Process domain tool calls first
    while hasattr(response, "tool_calls") and response.tool_calls:
        submit_call = None
        domain_calls = []

        for tc in response.tool_calls:
            if tc["name"] == "submit_triage_result":
                submit_call = tc
            else:
                domain_calls.append(tc)

        if submit_call:
            # Extract structured data from the tool call args
            print("    LLM called submit_triage_result with args:")
            args = submit_call["args"]

            try:
                result = TriageResult(**args)
                print(f"    Type: {type(result).__name__}")
                print(f"    Urgency          : {result.urgency}")
                print(f"    Risk score       : {result.risk_score:.2f}")
                print(f"    Primary concern  : {result.primary_concern}")
                print(f"    Findings ({len(result.findings)}):")
                for f in result.findings:
                    print(f"      - {f}")
                print(f"    Actions ({len(result.recommended_actions)}):")
                for a in result.recommended_actions:
                    print(f"      - {a}")
                print(f"    Confidence       : {result.confidence:.0%}")
                return result
            except ValidationError as e:
                print(f"    Validation failed: {e}")
                return None

        if domain_calls:
            print(f"    | Domain tool calls: {[tc['name'] for tc in domain_calls]}")
            tool_node = ToolNode(domain_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = bound_llm.invoke(messages, config=config)

    print("    LLM did not call submit_triage_result.")
    return None


# ============================================================
# STAGE 8.4 — Validation Demo
# ============================================================

def demonstrate_validation():
    """
    Show what happens when Pydantic catches malformed output.
    """
    print("\n\n    [STAGE 8.4] Pydantic validation")
    print("    " + "-" * 50)

    # Valid
    print("\n    Test 1: Valid data")
    try:
        result = TriageResult(
            urgency="urgent",
            risk_score=0.75,
            primary_concern="Hyperkalemia risk",
            findings=["K+ 5.4 mEq/L", "On dual K-raising agents"],
            recommended_actions=["ECG stat", "Hold Spironolactone"],
            confidence=0.85,
        )
        print(f"      -> OK: {result.urgency}, risk={result.risk_score}")
    except ValidationError as e:
        print(f"      -> FAIL: {e}")

    # Invalid urgency
    print("\n    Test 2: Invalid urgency value")
    try:
        result = TriageResult(
            urgency="critical",  # not in the Literal
            risk_score=0.9,
            primary_concern="Test",
            findings=["A"],
            recommended_actions=["B"],
            confidence=0.5,
        )
        print(f"      -> OK: {result.urgency}")
    except ValidationError as e:
        error_msg = str(e).split("\n")[1] if "\n" in str(e) else str(e)
        print(f"      -> FAIL (expected): {error_msg[:80]}")

    # Risk score out of range
    print("\n    Test 3: Risk score > 1.0")
    try:
        result = TriageResult(
            urgency="urgent",
            risk_score=1.5,  # out of range
            primary_concern="Test",
            findings=["A"],
            recommended_actions=["B"],
            confidence=0.5,
        )
        print(f"      -> OK: risk={result.risk_score}")
    except ValidationError as e:
        error_msg = str(e).split("\n")[1] if "\n" in str(e) else str(e)
        print(f"      -> FAIL (expected): {error_msg[:80]}")

    print("\n    Key point: Pydantic catches type errors, range errors,")
    print("    and enum violations BEFORE your code processes the data.")


# ============================================================
# STAGE 8.5 — Summary
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  STRUCTURED OUTPUT")
    print("  Pattern: get validated JSON from agents")
    print("=" * 70)

    print("""
    Two patterns for structured agent output:

    Pattern 1: with_structured_output()
      llm.with_structured_output(TriageResult)
      -> LLM returns a TriageResult object directly
      -> Cleanest approach, single call

    Pattern 2: Response-format tool
      @tool submit_triage_result(urgency, risk_score, ...)
      -> LLM calls this tool, args are the structured data
      -> Use when you also need domain tool calls first
    """)

    result_p1 = run_pattern_1()
    result_p2 = run_pattern_2()
    demonstrate_validation()

    print("\n\n" + "=" * 70)
    print("  STRUCTURED OUTPUT COMPLETE")
    print("=" * 70)
    print("""
    What you saw:
      Pattern 1: with_structured_output() returned a Pydantic object.
      Pattern 2: submit tool args were extracted as structured data.
      Validation: Pydantic caught invalid urgency and out-of-range scores.

    When to use which:
      Pattern 1: when the agent's ONLY job is to produce structured output.
      Pattern 2: when the agent needs to call domain tools first,
                 then submit findings in a structured format.

    Production tip:
      Always define your output schema as a Pydantic model.
      Types, constraints, and descriptions serve as documentation
      AND runtime validation AND LLM guidance.

    Next: dynamic_tool_selection.py — choosing tools at runtime.
    """)


if __name__ == "__main__":
    main()
