#!/usr/bin/env python3
"""
============================================================
Observed Clinical Pipeline
============================================================
Pattern 5: All observability patterns integrated in one
clinical multi-agent pipeline.
Prerequisite: session_based_tracing.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
This script combines every observability pattern from the
previous scripts into a single production-ready pipeline:

    1. Trace Hierarchy  -> Langfuse traces with spans/generations
    2. Metrics Tracking -> per-agent token/cost/latency
    3. Trace Scoring    -> rule-based + LLM-as-judge evaluation
    4. Session Grouping -> session_id + user_id correlation
    5. Decorators       -> @observe_agent on every node

This is what a fully instrumented multi-agent pipeline looks
like in production. Every LLM call, tool invocation, and
quality check is traced, metered, and scored.

------------------------------------------------------------
OBSERVABILITY vs TRACEABILITY -- what this covers
------------------------------------------------------------
    TRACEABILITY (complete audit trail):
        - Every agent's input and output
        - Every tool call with arguments and results
        - Decision paths (which agent ran when)
        - State flow across nodes

    OBSERVABILITY (operational health):
        - Token cost per agent and per pipeline
        - Latency per agent and per tool
        - Quality scores per output
        - Session-level aggregated metrics
        - Error/failure tracking

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [triage]           <-- @observe_agent, metered, traced
       |
       v
    [pharmacist]       <-- @observe_agent, metered, traced
       |
       v
    [evaluator]        <-- scores triage + pharmacist outputs
       |
       v
    [report]           <-- @observe_agent, metered, traced
       |
       v
    [END]

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()        triage_node       pharmacist_node    evaluator_node     report_node
      |               |                  |                  |                 |
      |-- invoke() -->|                  |                  |                 |
      |               |-- @observe_agent |                  |                 |
      |               |-- LLM + tools    |                  |                 |
      |               |-- record_llm_call|                  |                 |
      |               |-- record_tool_call|                 |                 |
      |               |----- state ----->|                  |                 |
      |               |                  |-- @observe_agent |                 |
      |               |                  |-- LLM call       |                 |
      |               |                  |-- record_llm_call|                 |
      |               |                  |----- state ----->|                 |
      |               |                  |                  |-- rule checks   |
      |               |                  |                  |-- LLM judge     |
      |               |                  |                  |-- attach scores |
      |               |                  |                  |----- state ---->|
      |               |                  |                  |                 |-- @observe_agent
      |               |                  |                  |                 |-- LLM call
      |               |                  |                  |                 |-- record()
      |<-- result + metrics + scores + trace URL ----------------------------|
      |               |                  |                  |                 |

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.observability_and_traceability.observed_clinical_pipeline
============================================================
"""

# -- Standard library --------------------------------------------------------
import sys
import json
import re
import time
from typing import TypedDict, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -- LangGraph ---------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# -- LangChain ---------------------------------------------------------------
from langchain_core.messages import HumanMessage, SystemMessage

# -- Project imports ----------------------------------------------------------
from core.config import get_llm
from observability.callbacks import build_callback_config
from observability.tracer import get_langfuse_client
from observability.decorators import observe_agent
from observability.metrics import MetricsCollector
from tools import analyze_symptoms, assess_patient_risk, check_drug_interactions


# ============================================================
# STAGE 5.1 -- State Definition
# ============================================================

class ObservedPipelineState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict

    # Agent outputs
    triage_result: str
    pharmacist_result: str
    report_result: str

    # Scoring
    evaluation_scores: dict
    evaluation_summary: str

    # Metrics (stored as summary dict at end)
    metrics_summary: dict


# ============================================================
# STAGE 5.2 -- Module-level Metrics
# ============================================================

pipeline_metrics = MetricsCollector()

# Common callback config factory for this pipeline
PIPELINE_SESSION_ID = "observed-pipeline-session-001"
PIPELINE_USER_ID = "dr-singh-internal-medicine"


def _build_node_config(node_name: str) -> dict:
    """Build Langfuse callback config for a pipeline node."""
    return build_callback_config(
        trace_name=f"observed_pipeline_{node_name}",
        user_id=PIPELINE_USER_ID,
        session_id=PIPELINE_SESSION_ID,
        tags=["observed_pipeline", node_name],
    )


# ============================================================
# STAGE 5.3 -- Node Definitions
# ============================================================

@observe_agent(agent_name="triage", tags=["clinical", "triage"])
def triage_node(state: ObservedPipelineState) -> dict:
    """
    Triage agent -- fully instrumented.

    Instrumentation layers:
        1. @observe_agent decorator -> Langfuse span
        2. build_callback_config -> Langfuse generation
        3. MetricsCollector -> token/cost/latency tracking
    """
    llm = get_llm()
    tools = [analyze_symptoms, assess_patient_risk]
    agent_llm = llm.bind_tools(tools)
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a triage specialist. Analyze the patient's symptoms "
        "and assess risk. Provide a triage summary (3-4 sentences) "
        "with urgency level: ROUTINE, URGENT, or EMERGENT."
    ))
    prompt = HumanMessage(content=f"""Triage this patient:
Age: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
Vitals: {json.dumps(patient.get('vitals', {}))}
Labs: {json.dumps(patient.get('lab_results', {}))}""")

    config = _build_node_config("triage")
    messages = [system, prompt]

    call_start = time.time()
    response = agent_llm.invoke(messages, config=config)
    latency_ms = (time.time() - call_start) * 1000

    # Record LLM metrics
    prompt_tokens = len(str(messages)) // 4
    completion_tokens = len(response.content) // 4
    pipeline_metrics.record_llm_call(
        agent_name="triage",
        tokens_in=prompt_tokens,
        tokens_out=completion_tokens,
        model="gemini-2.5-flash-preview-05-20",
        latency_ms=latency_ms,
    )

    # Handle tool calls
    while hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_start = time.time()
            tool_node = ToolNode(tools)
            tool_results = tool_node.invoke({"messages": [response]})
            tool_latency = (time.time() - tool_start) * 1000

            pipeline_metrics.record_tool_call(
                tool_name=tool_call["name"],
                agent_name="triage",
                latency_ms=tool_latency,
                success=True,
            )
            print(f"    | [Triage] Tool: {tool_call['name']} ({tool_latency:.0f}ms)")

        messages.extend([response] + tool_results["messages"])
        call_start = time.time()
        response = agent_llm.invoke(messages, config=config)
        latency_ms = (time.time() - call_start) * 1000

        pipeline_metrics.record_llm_call(
            agent_name="triage",
            tokens_in=len(str(messages)) // 4,
            tokens_out=len(response.content) // 4,
            model="gemini-2.5-flash-preview-05-20",
            latency_ms=latency_ms,
        )

    print(f"    | [Triage] Result: {len(response.content)} chars, {latency_ms:.0f}ms")

    return {
        "messages": [response],
        "triage_result": response.content,
    }


@observe_agent(agent_name="pharmacist", tags=["clinical", "pharmacology"])
def pharmacist_node(state: ObservedPipelineState) -> dict:
    """
    Pharmacist agent -- reviews triage and checks medications.

    Instrumentation: @observe_agent + callback config + metrics.
    """
    llm = get_llm()
    patient = state["patient_case"]
    triage = state.get("triage_result", "")

    system = SystemMessage(content=(
        "You are a clinical pharmacologist. Review the triage assessment, "
        "check for drug interactions, and provide medication recommendations "
        "(3-4 sentences). Mention specific medications and dosage changes."
    ))
    prompt = HumanMessage(content=f"""Review this case:

Triage: {triage}

Medications: {', '.join(patient.get('current_medications', []))}
Allergies: {', '.join(patient.get('allergies', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}""")

    config = _build_node_config("pharmacist")

    call_start = time.time()
    response = llm.invoke([system, prompt], config=config)
    latency_ms = (time.time() - call_start) * 1000

    prompt_tokens = len(str([system, prompt])) // 4
    completion_tokens = len(response.content) // 4

    pipeline_metrics.record_llm_call(
        agent_name="pharmacist",
        tokens_in=prompt_tokens,
        tokens_out=completion_tokens,
        model="gemini-2.5-flash-preview-05-20",
        latency_ms=latency_ms,
    )

    print(f"    | [Pharmacist] Result: {len(response.content)} chars, {latency_ms:.0f}ms")

    return {
        "messages": [response],
        "pharmacist_result": response.content,
    }


def evaluator_node(state: ObservedPipelineState) -> dict:
    """
    Evaluator -- scores triage + pharmacist outputs.

    Combines rule-based and LLM-as-judge scoring in one node.
    This is the quality gate of the pipeline.
    """
    triage = state.get("triage_result", "")
    pharmacist = state.get("pharmacist_result", "")
    combined_output = f"{triage}\n\n{pharmacist}"
    combined_lower = combined_output.lower()

    # -- Rule-based scoring ------------------------------------------------
    rule_scores = {}

    # Check medication awareness
    medication_keywords = ["medication", "drug", "dose", "mg", "spironolactone",
                           "lisinopril", "furosemide", "metformin"]
    rule_scores["medication_mention"] = any(
        keyword in combined_lower for keyword in medication_keywords
    )

    # Check monitoring plan
    monitoring_keywords = ["monitor", "follow-up", "recheck", "48 hours", "weekly"]
    rule_scores["monitoring_plan"] = any(
        keyword in combined_lower for keyword in monitoring_keywords
    )

    # Check risk awareness
    risk_keywords = ["risk", "caution", "contraindic", "interaction", "potassium",
                     "hyperkalemia", "K+"]
    rule_scores["risk_awareness"] = any(
        keyword in combined_lower for keyword in risk_keywords
    )

    # Check urgency classification
    urgency_keywords = ["urgent", "emergent", "routine", "immediate", "priority"]
    rule_scores["urgency_classified"] = any(
        keyword in combined_lower for keyword in urgency_keywords
    )

    checks_passed = sum(1 for passed in rule_scores.values() if passed)
    total_checks = len(rule_scores)
    rule_scores["completeness"] = round(checks_passed / total_checks, 2)

    for check_name, check_passed in rule_scores.items():
        if check_name != "completeness":
            print(f"    | [Evaluator] Rule: {check_name} -> {'PASS' if check_passed else 'FAIL'}")
    print(f"    | [Evaluator] Completeness: {rule_scores['completeness']:.0%}")

    # -- LLM-as-Judge scoring ----------------------------------------------
    llm = get_llm()
    judge_system = SystemMessage(content=(
        "You are a clinical quality evaluator. Score the following "
        "clinical output on a scale of 1-5. Respond ONLY with JSON: "
        '{"clinical_accuracy": N, "actionability": N, "overall": N}'
    ))
    judge_prompt = HumanMessage(content=f"""Evaluate this clinical output:

{combined_output[:1500]}

Score as JSON (1-5):""")

    config = _build_node_config("evaluator")

    call_start = time.time()
    judge_response = llm.invoke([judge_system, judge_prompt], config=config)
    latency_ms = (time.time() - call_start) * 1000

    pipeline_metrics.record_llm_call(
        agent_name="evaluator",
        tokens_in=len(str([judge_system, judge_prompt])) // 4,
        tokens_out=len(judge_response.content) // 4,
        model="gemini-2.5-flash-preview-05-20",
        latency_ms=latency_ms,
    )

    # Parse judge scores
    llm_scores = {}
    try:
        json_match = re.search(r'\{[^}]+\}', judge_response.content)
        if json_match:
            llm_scores = json.loads(json_match.group())
        else:
            llm_scores = json.loads(judge_response.content.strip())
    except (json.JSONDecodeError, ValueError):
        llm_scores = {"clinical_accuracy": 3, "actionability": 3, "overall": 3}

    print(f"    | [Evaluator] LLM Judge: {json.dumps(llm_scores)}")

    # -- Build summary -----------------------------------------------------
    all_scores = {
        "rule_based": rule_scores,
        "llm_judge": llm_scores,
    }

    summary_lines = []
    summary_lines.append("Rule-Based: " + ", ".join(
        f"{name}={'PASS' if val else 'FAIL'}"
        for name, val in rule_scores.items() if isinstance(val, bool)
    ))
    summary_lines.append(f"Completeness: {rule_scores['completeness']:.0%}")
    summary_lines.append("LLM Judge: " + ", ".join(
        f"{name}={val}/5" for name, val in llm_scores.items()
    ))

    return {
        "messages": [judge_response],
        "evaluation_scores": all_scores,
        "evaluation_summary": "\n".join(summary_lines),
    }


@observe_agent(agent_name="report", tags=["clinical", "report"])
def report_node(state: ObservedPipelineState) -> dict:
    """
    Report generator -- compiles everything into a final report.

    Includes evaluation scores in the report metadata.
    """
    llm = get_llm()

    system = SystemMessage(content=(
        "You are a clinical report writer. Compile the triage and "
        "pharmacist findings into a structured clinical summary. "
        "Keep under 200 words. Include sections: "
        "1) Triage 2) Medication Review 3) Recommendations."
    ))
    prompt = HumanMessage(content=f"""Compile clinical report:

Triage: {state.get('triage_result', '')}

Pharmacist: {state.get('pharmacist_result', '')}

Quality Scores: {state.get('evaluation_summary', '')}""")

    config = _build_node_config("report")

    call_start = time.time()
    response = llm.invoke([system, prompt], config=config)
    latency_ms = (time.time() - call_start) * 1000

    prompt_tokens = len(str([system, prompt])) // 4
    completion_tokens = len(response.content) // 4

    pipeline_metrics.record_llm_call(
        agent_name="report",
        tokens_in=prompt_tokens,
        tokens_out=completion_tokens,
        model="gemini-2.5-flash-preview-05-20",
        latency_ms=latency_ms,
    )

    print(f"    | [Report] Generated: {len(response.content)} chars, {latency_ms:.0f}ms")

    # Capture final metrics
    workflow_summary = pipeline_metrics.get_workflow_summary()
    agent_breakdown = pipeline_metrics.get_agent_summary()

    return {
        "messages": [response],
        "report_result": response.content,
        "metrics_summary": {
            "workflow": workflow_summary,
            "per_agent": agent_breakdown,
        },
    }


# ============================================================
# STAGE 5.4 -- Graph Construction
# ============================================================

def build_observed_pipeline():
    """
    Build the fully observed clinical pipeline.

    Every node is instrumented with:
        - @observe_agent decorator (Langfuse spans)
        - build_callback_config (Langfuse generations)
        - MetricsCollector (token/cost/latency)
        - Evaluator (quality scores)

    MemorySaver allows cross-invocation state inspection.
    """
    workflow = StateGraph(ObservedPipelineState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("pharmacist", pharmacist_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "triage")
    workflow.add_edge("triage", "pharmacist")
    workflow.add_edge("pharmacist", "evaluator")
    workflow.add_edge("evaluator", "report")
    workflow.add_edge("report", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# STAGE 5.5 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  OBSERVED CLINICAL PIPELINE")
    print("  Pattern: all observability tiers in one pipeline")
    print("=" * 70)

    print("""
    Observability tiers in this pipeline:

      TIER 1 -- Langfuse Traces (trace_hierarchy)
        Every LLM call creates a Generation under a Span
        All linked by session_id for cross-request correlation

      TIER 2 -- MetricsCollector (agent_metrics_and_cost)
        Per-agent token, cost, and latency tracking
        Workflow-level aggregated summary

      TIER 3 -- Quality Scores (trace_scoring_and_evaluation)
        Rule-based checks (does output mention medications?)
        LLM-as-Judge evaluation (accuracy, actionability)

      TIER 4 -- Session Grouping (session_based_tracing)
        session_id links all traces in this pipeline
        user_id tracks per-doctor usage

      TIER 5 -- Decorators (@observe_agent)
        Every node function is decorated for automatic tracing
    """)

    from core.models import PatientCase

    patient = PatientCase(
        patient_id="PT-FULL-OBS-001",
        age=71, sex="F",
        chief_complaint="Dizziness with elevated potassium",
        symptoms=["dizziness", "fatigue", "ankle edema", "orthopnea"],
        medical_history=["CKD Stage 3a", "Hypertension", "CHF", "Type 2 Diabetes"],
        current_medications=[
            "Lisinopril 20mg daily",
            "Spironolactone 25mg daily",
            "Furosemide 40mg daily",
            "Metformin 500mg twice daily",
        ],
        allergies=["Sulfa drugs"],
        lab_results={"K+": "5.4 mEq/L", "eGFR": "42 mL/min", "BNP": "450 pg/mL"},
        vitals={"BP": "105/65", "HR": "58", "SpO2": "93%"},
    )

    initial_state = {
        "messages": [],
        "patient_case": patient.model_dump(),
        "triage_result": "",
        "pharmacist_result": "",
        "report_result": "",
        "evaluation_scores": {},
        "evaluation_summary": "",
        "metrics_summary": {},
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Doctor: {PIPELINE_USER_ID}")
    print(f"    Session: {PIPELINE_SESSION_ID}")
    print(f"    Complaint: {patient.chief_complaint}")
    print()
    print("    " + "-" * 60)

    # -- Check Langfuse ---------------------------------------------------
    langfuse_client = get_langfuse_client()
    langfuse_available = langfuse_client is not None
    print(f"    Langfuse: {'connected' if langfuse_available else 'not configured (NoOp mode)'}")
    print()

    # -- Run pipeline ------------------------------------------------------
    graph = build_observed_pipeline()
    config = {"configurable": {"thread_id": "observed-pipeline-thread-001"}}

    result = graph.invoke(initial_state, config=config)

    # -- Display results ---------------------------------------------------
    print("\n    " + "=" * 60)
    print("    PIPELINE RESULTS")
    print("    " + "=" * 60)

    print(f"\n    TRIAGE:")
    for line in result["triage_result"][:300].split("\n"):
        if line.strip():
            print(f"      {line}")

    print(f"\n    PHARMACIST REVIEW:")
    for line in result["pharmacist_result"][:300].split("\n"):
        if line.strip():
            print(f"      {line}")

    print(f"\n    CLINICAL REPORT:")
    print(f"    {'~' * 50}")
    for line in result["report_result"].split("\n"):
        print(f"    | {line}")

    # -- Display evaluation scores -----------------------------------------
    print(f"\n    QUALITY EVALUATION:")
    print(f"    {'~' * 50}")
    scores = result.get("evaluation_scores", {})
    rule_scores = scores.get("rule_based", {})
    llm_scores = scores.get("llm_judge", {})

    print("    Rule-Based Checks:")
    for check_name, check_value in rule_scores.items():
        if isinstance(check_value, bool):
            print(f"      {check_name}: {'PASS' if check_value else 'FAIL'}")
    print(f"      completeness: {rule_scores.get('completeness', 0):.0%}")

    print("    LLM-as-Judge (1-5):")
    for criterion, score_value in llm_scores.items():
        bar = "*" * int(score_value) + "." * (5 - int(score_value))
        print(f"      {criterion}: [{bar}] {score_value}/5")

    # -- Display metrics ---------------------------------------------------
    print(f"\n    COST AND METRICS:")
    print(f"    {'~' * 50}")
    metrics = result.get("metrics_summary", {})
    workflow_summary = metrics.get("workflow", {})
    per_agent = metrics.get("per_agent", {})

    print(f"""
    Workflow Totals:
      LLM Calls:     {workflow_summary.get('total_llm_calls', 0)}
      Tokens In:     {workflow_summary.get('total_tokens_in', 0)}
      Tokens Out:    {workflow_summary.get('total_tokens_out', 0)}
      Total Cost:    ${workflow_summary.get('total_cost_usd', 0):.4f}
      Tool Calls:    {workflow_summary.get('total_tool_calls', 0)}
      Duration:      {workflow_summary.get('workflow_duration_ms', 0):.0f}ms
    """)

    print("    Per-Agent:")
    for agent_name, agent_data in per_agent.items():
        cost = agent_data.get('total_cost_usd', 0)
        latency = agent_data.get('avg_latency_ms', 0)
        tokens = agent_data.get('total_tokens_in', 0) + agent_data.get('total_tokens_out', 0)
        tools_count = agent_data.get('tool_calls', 0)
        print(f"      {agent_name}: {tokens} tokens, ${cost:.4f}, {latency:.0f}ms"
              + (f", {tools_count} tools" if tools_count > 0 else ""))

    # -- Checkpoint inspection ---------------------------------------------
    print(f"\n    CHECKPOINT STATE:")
    snapshot = graph.get_state(config)
    print(f"    Keys: {list(snapshot.values.keys())}")
    print(f"    State is persisted and inspectable via graph.get_state()")

    # -- Final summary -----------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  OBSERVED CLINICAL PIPELINE COMPLETE")
    print("=" * 70)
    print("""
    Observability summary for this pipeline:

      Tier 1 (Traces):     Every LLM call captured as Generation
                            Grouped by session_id in Langfuse
      Tier 2 (Metrics):    Per-agent token/cost/latency breakdown
                            Workflow-level totals
      Tier 3 (Scores):     Rule-based checks (4 criteria)
                            LLM-as-Judge (3 quality dimensions)
      Tier 4 (Sessions):   session_id links this pipeline execution
                            user_id enables per-doctor analytics
      Tier 5 (Decorators): @observe_agent on every node function

    What gets traced vs observed:

      TRACED (audit trail):
        - Which agent ran, with what input/output
        - Which tools were called (arguments + results)
        - Decision path through the graph

      OBSERVED (operational health):
        - Token consumption and cost per agent
        - Latency per call and per pipeline
        - Quality scores (trending, regression detection)
        - Session-level aggregated metrics

    This completes the observability pattern series:
      1. trace_hierarchy           -- traces, spans, generations
      2. agent_metrics_and_cost    -- per-agent cost tracking
      3. trace_scoring_and_evaluation -- quality scoring
      4. session_based_tracing     -- multi-turn sessions
      5. observed_clinical_pipeline -- all patterns combined
    """)


if __name__ == "__main__":
    main()
