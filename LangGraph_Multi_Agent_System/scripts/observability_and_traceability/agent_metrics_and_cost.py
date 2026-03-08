 #!/usr/bin/env python3
"""
============================================================
Agent Metrics and Cost Tracking
============================================================
Pattern 2: Per-agent token usage, latency, and cost tracking
using MetricsCollector in a multi-agent pipeline.
Prerequisite: trace_hierarchy.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
In production multi-agent systems, you need to know:

    1. Which agent consumes the most tokens (cost driver)?
    2. Which agent is the slowest (latency bottleneck)?
    3. Which tools fail most often (reliability risk)?
    4. What is the total cost per workflow execution?

MetricsCollector tracks all of this. This script shows how
to instrument a LangGraph pipeline to record metrics at
every node, then display a per-agent cost breakdown.

------------------------------------------------------------
OBSERVABILITY vs TRACEABILITY — what this covers
------------------------------------------------------------
    TRACEABILITY:
        - Which agent made which LLM call
        - Which tools were invoked by each agent

    OBSERVABILITY:
        - Token consumption per agent (input + output)
        - Cost per agent (model-specific pricing)
        - Latency per LLM call and per tool call
        - Tool success/failure rates
        - Workflow-level totals and per-agent averages

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [triage]           <-- LLM call + tool calls, metrics recorded
       |
       v
    [drug_check]       <-- tool-only node, metrics recorded
       |
       v
    [pharmacist]       <-- LLM call + tool calls, metrics recorded
       |
       v
    [report]           <-- LLM call, metrics recorded
       |
       v
    [END]

    MetricsCollector is carried in state and updated by each node.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()        triage_node         drug_check_node      pharmacist_node     report_node
      |               |                    |                    |                  |
      |-- invoke() -->|                    |                    |                  |
      |               |-- LLM call         |                    |                  |
      |               |-- record_llm_call()|                    |                  |
      |               |-- tool call        |                    |                  |
      |               |-- record_tool_call()|                   |                  |
      |               |----- state ------->|                    |                  |
      |               |                    |-- tool calls only  |                  |
      |               |                    |-- record_tool_call()|                 |
      |               |                    |----- state ------->|                  |
      |               |                    |                    |-- LLM call       |
      |               |                    |                    |-- record_llm_call()|
      |               |                    |                    |----- state ------>|
      |               |                    |                    |                   |-- LLM call
      |               |                    |                    |                   |-- record()
      |<-- result + metrics_summary ------------------------------------------------|
      |               |                    |                    |                  |

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.observability_and_traceability.agent_metrics_and_cost
============================================================
"""

# -- Standard library --------------------------------------------------------
import sys
import json
import time
from typing import TypedDict, Annotated, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# -- LangGraph ---------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# -- LangChain ---------------------------------------------------------------
from langchain_core.messages import HumanMessage, SystemMessage

# -- Project imports ----------------------------------------------------------
# CONNECTION: core/ root module — get_llm() centralises LLM config so all nodes
# use the same provider/model without hardcoding credentials.
from core.config import get_llm
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse tracing to every LLM call. MetricsCollector (observability/metrics.py)
# is the component that tracks per-agent token usage, latency, and cost.
# This script demonstrates HOW to instrument a pipeline with MetricsCollector
# — the pattern, not the MetricsCollector implementation itself.
from observability.callbacks import build_callback_config
from observability.metrics import MetricsCollector
# CONNECTION: tools/ root module — the clinical tool functions used by agents.
# This script records tool call metrics (latency, success/failure) on top of them.
from tools import analyze_symptoms, assess_patient_risk, check_drug_interactions


# ============================================================
# STAGE 2.1 -- State Definition
# ============================================================

class MetricsTrackingState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_case: dict
    triage_result: str
    drug_check_result: str
    pharmacist_result: str
    report_result: str
    # Metrics are tracked externally (MetricsCollector instance)
    # because dataclass instances cannot be serialised into state.
    # We store the summary dict at the end for display.
    metrics_summary: dict


# ============================================================
# STAGE 2.2 -- Module-level MetricsCollector
# ============================================================
# MetricsCollector is a mutable object. We create it at module
# level so all node functions can access it. In production,
# you would pass it via dependency injection or context.

workflow_metrics = MetricsCollector()


# ============================================================
# STAGE 2.3 -- Node Definitions
# ============================================================

def triage_node(state: MetricsTrackingState) -> dict:
    """
    Triage agent -- LLM call with tool usage, fully metered.

    Demonstrates:
        - Recording an LLM call (tokens, model, latency)
        - Recording tool calls (latency, success/failure)
    """
    llm = get_llm()
    tools = [analyze_symptoms, assess_patient_risk]
    agent_llm = llm.bind_tools(tools)
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a triage specialist. Use your tools to analyze the "
        "patient's symptoms and assess risk level. Provide a triage "
        "summary (3-4 sentences) with urgency classification."
    ))
    prompt = HumanMessage(content=f"""Triage this patient:
Age: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
Vitals: {json.dumps(patient.get('vitals', {}))}
Labs: {json.dumps(patient.get('lab_results', {}))}""")

    config = build_callback_config(trace_name="metrics_triage")
    messages = [system, prompt]

    # -- Timed LLM call ----------------------------------------------------
    call_start = time.time()
    response = agent_llm.invoke(messages, config=config)
    llm_latency_ms = (time.time() - call_start) * 1000

    # Record LLM call metrics (estimated tokens from response length)
    prompt_token_estimate = len(str(messages)) // 4
    completion_token_estimate = len(response.content) // 4
    workflow_metrics.record_llm_call(
        agent_name="triage",
        tokens_in=prompt_token_estimate,
        tokens_out=completion_token_estimate,
        model="gemini-2.5-flash-preview-05-20",
        latency_ms=llm_latency_ms,
    )
    print(f"    | [Triage] LLM: {prompt_token_estimate}+{completion_token_estimate} tokens, {llm_latency_ms:.0f}ms")

    # -- Handle tool calls -------------------------------------------------
    while hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_start = time.time()
            tool_name = tool_call["name"]
            print(f"    | [Triage] Tool: {tool_name}")

            tool_node = ToolNode(tools)
            tool_results = tool_node.invoke({"messages": [response]})
            tool_latency_ms = (time.time() - tool_start) * 1000

            workflow_metrics.record_tool_call(
                tool_name=tool_name,
                agent_name="triage",
                latency_ms=tool_latency_ms,
                success=True,
            )
            print(f"    | [Triage] Tool {tool_name}: {tool_latency_ms:.0f}ms")

        messages.extend([response] + tool_results["messages"])
        call_start = time.time()
        response = agent_llm.invoke(messages, config=config)
        llm_latency_ms = (time.time() - call_start) * 1000

        completion_token_estimate = len(response.content) // 4
        workflow_metrics.record_llm_call(
            agent_name="triage",
            tokens_in=len(str(messages)) // 4,
            tokens_out=completion_token_estimate,
            model="gemini-2.5-flash-preview-05-20",
            latency_ms=llm_latency_ms,
        )

    return {
        "messages": [response],
        "triage_result": response.content,
    }


def drug_check_node(state: MetricsTrackingState) -> dict:
    """
    Drug interaction check -- tool-only node (no LLM).

    Demonstrates recording tool metrics without LLM calls.
    This is common in pipelines where some nodes are pure
    tool execution (database lookups, API calls, calculations).
    """
    patient = state["patient_case"]
    medications = patient.get("current_medications", [])

    print(f"    | [DrugCheck] Checking {len(medications)} medications")

    # Simulate checking each pair of medications
    checked_pairs = []
    for index, medication in enumerate(medications):
        tool_start = time.time()
        try:
            # Use the check_drug_interactions tool
            result = check_drug_interactions.invoke({
                "drug_a": medication,
                "drug_b": medications[(index + 1) % len(medications)],
            })
            tool_latency_ms = (time.time() - tool_start) * 1000
            workflow_metrics.record_tool_call(
                tool_name="check_drug_interactions",
                agent_name="drug_check",
                latency_ms=tool_latency_ms,
                success=True,
            )
            checked_pairs.append(result)
            print(f"    | [DrugCheck] Pair {index + 1}: {tool_latency_ms:.0f}ms")
        except Exception as error:
            tool_latency_ms = (time.time() - tool_start) * 1000
            workflow_metrics.record_tool_call(
                tool_name="check_drug_interactions",
                agent_name="drug_check",
                latency_ms=tool_latency_ms,
                success=False,
            )
            print(f"    | [DrugCheck] Pair {index + 1}: FAILED ({error})")

    drug_check_summary = f"Checked {len(checked_pairs)} drug pairs. Results: {json.dumps(checked_pairs[:3])}"

    return {
        "drug_check_result": drug_check_summary,
    }


def pharmacist_node(state: MetricsTrackingState) -> dict:
    """
    Pharmacist agent -- LLM call with context from upstream agents.

    Uses triage result + drug check result + guidelines.
    Typically the most expensive agent (long prompts, complex reasoning).
    """
    llm = get_llm()
    patient = state["patient_case"]
    triage = state.get("triage_result", "")
    drug_check = state.get("drug_check_result", "")

    system = SystemMessage(content=(
        "You are a clinical pharmacologist. Review the triage assessment "
        "and drug interaction check results. Provide a medication review "
        "(3-4 sentences) with specific recommendations."
    ))
    prompt = HumanMessage(content=f"""Review this case:

Triage: {triage}

Drug Check: {drug_check}

Patient Medications: {', '.join(patient.get('current_medications', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}""")

    config = build_callback_config(trace_name="metrics_pharmacist")

    call_start = time.time()
    response = llm.invoke([system, prompt], config=config)
    llm_latency_ms = (time.time() - call_start) * 1000

    prompt_token_estimate = len(str([system, prompt])) // 4
    completion_token_estimate = len(response.content) // 4

    workflow_metrics.record_llm_call(
        agent_name="pharmacist",
        tokens_in=prompt_token_estimate,
        tokens_out=completion_token_estimate,
        model="gemini-2.5-flash-preview-05-20",
        latency_ms=llm_latency_ms,
    )
    print(f"    | [Pharmacist] LLM: {prompt_token_estimate}+{completion_token_estimate} tokens, {llm_latency_ms:.0f}ms")

    return {
        "messages": [response],
        "pharmacist_result": response.content,
    }


def report_node(state: MetricsTrackingState) -> dict:
    """
    Report generator -- final LLM call, lowest cost.

    Compiles findings from all upstream agents into a report.
    """
    llm = get_llm()

    system = SystemMessage(content=(
        "You are a clinical report writer. Compile findings from "
        "triage, drug check, and pharmacist review into a structured "
        "clinical summary. Keep under 150 words."
    ))
    prompt = HumanMessage(content=f"""Compile clinical report:

Triage: {state.get('triage_result', '')}

Drug Check: {state.get('drug_check_result', '')}

Pharmacist Review: {state.get('pharmacist_result', '')}""")

    config = build_callback_config(trace_name="metrics_report")

    call_start = time.time()
    response = llm.invoke([system, prompt], config=config)
    llm_latency_ms = (time.time() - call_start) * 1000

    prompt_token_estimate = len(str([system, prompt])) // 4
    completion_token_estimate = len(response.content) // 4

    workflow_metrics.record_llm_call(
        agent_name="report",
        tokens_in=prompt_token_estimate,
        tokens_out=completion_token_estimate,
        model="gemini-2.5-flash-preview-05-20",
        latency_ms=llm_latency_ms,
    )
    print(f"    | [Report] LLM: {prompt_token_estimate}+{completion_token_estimate} tokens, {llm_latency_ms:.0f}ms")

    # Capture the final metrics summary into state
    summary = workflow_metrics.get_workflow_summary()
    agent_breakdown = workflow_metrics.get_agent_summary()

    return {
        "messages": [response],
        "report_result": response.content,
        "metrics_summary": {
            "workflow": summary,
            "per_agent": agent_breakdown,
        },
    }


# ============================================================
# STAGE 2.4 -- Graph Construction
# ============================================================

def build_metrics_graph():
    """Build the 4-node metered pipeline."""
    workflow = StateGraph(MetricsTrackingState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("drug_check", drug_check_node)
    workflow.add_node("pharmacist", pharmacist_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "triage")
    workflow.add_edge("triage", "drug_check")
    workflow.add_edge("drug_check", "pharmacist")
    workflow.add_edge("pharmacist", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# ============================================================
# STAGE 2.5 -- Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  AGENT METRICS AND COST TRACKING")
    print("  Pattern: per-agent token, latency, and cost instrumentation")
    print("=" * 70)

    print("""
    MetricsCollector tracks three categories:

      1. LLM CALLS — per agent:
         - Token count (input + output)
         - Model used (for cost calculation)
         - Latency (response time in ms)
         - Cost (calculated from model pricing table)

      2. TOOL CALLS — per agent:
         - Tool name and invocation count
         - Latency per call
         - Success/failure status

      3. WORKFLOW TOTALS:
         - Total LLM calls, tokens, cost
         - Total tool calls, failure rate
         - End-to-end duration
    """)

    from core.models import PatientCase

    patient = PatientCase(
        patient_id="PT-OBS-001",
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
        "drug_check_result": "",
        "pharmacist_result": "",
        "report_result": "",
        "metrics_summary": {},
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print()
    print("    " + "-" * 60)

    graph = build_metrics_graph()
    result = graph.invoke(initial_state)

    # -- Display metrics breakdown -----------------------------------------
    print("\n    " + "=" * 60)
    print("    COST AND METRICS BREAKDOWN")
    print("    " + "=" * 60)

    metrics = result.get("metrics_summary", {})
    workflow_summary = metrics.get("workflow", {})
    per_agent = metrics.get("per_agent", {})

    # Per-agent table
    print(f"""
    Per-Agent Metrics:
    +------------------+--------+----------+----------+-----------+---------+
    | Agent            | LLM    | Tokens   | Tokens   | Cost      | Latency |
    |                  | Calls  | In       | Out      | (USD)     | Avg(ms) |
    +------------------+--------+----------+----------+-----------+---------+""")

    for agent_name, agent_data in per_agent.items():
        llm_calls = agent_data.get("llm_calls", 0)
        tokens_in = agent_data.get("total_tokens_in", 0)
        tokens_out = agent_data.get("total_tokens_out", 0)
        cost = agent_data.get("total_cost_usd", 0.0)
        avg_latency = agent_data.get("avg_latency_ms", 0.0)
        tool_calls = agent_data.get("tool_calls", 0)
        tool_fails = agent_data.get("tool_failures", 0)

        print(f"    | {agent_name:<16} | {llm_calls:>6} | {tokens_in:>8} | {tokens_out:>8} | ${cost:>8.4f} | {avg_latency:>7.0f} |")
        if tool_calls > 0:
            print(f"    |   tools: {tool_calls} calls, {tool_fails} failures{' ' * 36}|")

    print(f"    +------------------+--------+----------+----------+-----------+---------+")

    # Workflow totals
    total_llm = workflow_summary.get("total_llm_calls", 0)
    total_in = workflow_summary.get("total_tokens_in", 0)
    total_out = workflow_summary.get("total_tokens_out", 0)
    total_cost = workflow_summary.get("total_cost_usd", 0.0)
    total_tools = workflow_summary.get("total_tool_calls", 0)
    tool_failures = workflow_summary.get("tool_failure_count", 0)
    duration = workflow_summary.get("workflow_duration_ms", 0.0)

    print(f"""
    Workflow Totals:
      LLM Calls:     {total_llm}
      Total Tokens:  {total_in} in + {total_out} out = {total_in + total_out}
      Total Cost:    ${total_cost:.4f}
      Tool Calls:    {total_tools} ({tool_failures} failures)
      Duration:      {duration:.0f}ms
      Agents:        {', '.join(per_agent.keys())}
    """)

    # -- Report output -----------------------------------------------------
    print("    " + "=" * 60)
    print("    CLINICAL REPORT")
    print("    " + "-" * 60)
    for line in result["report_result"].split("\n"):
        print(f"    | {line}")

    # -- Summary -----------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  METRICS TRACKING SUMMARY")
    print("=" * 70)
    print("""
    What you learned:

      1. MetricsCollector.record_llm_call() captures:
         - agent_name, model, tokens_in, tokens_out, latency_ms
         - Cost is auto-calculated from model pricing table

      2. MetricsCollector.record_tool_call() captures:
         - tool_name, agent_name, latency_ms, success

      3. get_workflow_summary() aggregates across all agents
      4. get_agent_summary() breaks down per agent
      5. These metrics can be attached to Langfuse traces as metadata

    Why this matters:
      - Identify the most expensive agent (cost optimisation)
      - Find latency bottlenecks (performance tuning)
      - Track tool reliability (error rate monitoring)
      - Set token budgets per agent (cost control)

    Next: trace_scoring_and_evaluation.py -- quality scores on traces.
    """)


if __name__ == "__main__":
    main()
