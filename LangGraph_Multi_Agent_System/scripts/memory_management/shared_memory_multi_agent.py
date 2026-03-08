#!/usr/bin/env python3
"""
============================================================
Shared Memory Multi-Agent
============================================================
Pattern 5: All memory tiers combined in one clinical pipeline.
Integration script — combines WorkingMemory, checkpoints,
ChromaDB RAG, and conversation history.
Prerequisite: conversation_memory.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Real multi-agent systems use MULTIPLE memory tiers together.
This script demonstrates how each tier serves a distinct
purpose in the same pipeline:

    Tier 1: LangGraph State
        → carries patient_case, assessment, plan between nodes
        → scope: one invoke() call

    Tier 2: WorkingMemory (scratchpad)
        → triage writes findings, pharmacist reads them
        → scope: one pipeline execution

    Tier 3: Checkpoints (MemorySaver)
        → frozen state for HITL pause/resume
        → scope: per thread_id, until deleted

    Tier 4: Long-term memory (ChromaDB RAG)
        → drug guidelines retrieved at query time
        → scope: permanent, shared across all runs

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [triage]           <-- real LLM, writes to WorkingMemory
       |
       v
    [retrieve_guides]  <-- queries ChromaDB for guidelines
       |
       v
    [pharmacist]       <-- real LLM, reads WorkingMemory + guidelines
       |
       v
    [report]           <-- real LLM, reads full WorkingMemory, produces report
       |
       v
    [END]

------------------------------------------------------------
MEMORY FLOW DIAGRAM
------------------------------------------------------------

    ┌─────────────────────────────────────────────────────────┐
    │                    LangGraph State                       │
    │  patient_case ──┬──────────────────────────────────────  │
    │  messages ──────┤  flows through all nodes               │
    │  assessment ────┘                                        │
    ├─────────────────────────────────────────────────────────┤
    │              WorkingMemory (scratchpad)                  │
    │  triage → {findings, risk}                               │
    │  pharmacist → {warnings, meds_checked}                   │
    │  report → reads all + generates summary                  │
    ├─────────────────────────────────────────────────────────┤
    │              ChromaDB (long-term memory)                 │
    │  retrieve_guides → searches for condition guidelines     │
    │  pharmacist → uses retrieved context in prompt           │
    ├─────────────────────────────────────────────────────────┤
    │              MemorySaver (checkpoints)                   │
    │  state saved after every node                            │
    │  inspectable via graph.get_state(config)                 │
    └─────────────────────────────────────────────────────────┘

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()        triage_node      retrieve_guides     pharmacist_node     report_node
      |               |                  |                   |                  |
      |-- invoke() -->|                  |                   |                  |
      |               |-- LLM + tools    |                   |                  |
      |               |-- load_working_memory()              |                  |
      |               |-- set(triage_assessment)              |                  |
      |               |-- save_working_memory()               |                  |
      |               |----- state ----->|                   |                  |
      |               |                  |-- load_working_memory()             |
      |               |                  |-- search(ChromaDB)|                  |
      |               |                  |-- save_working_memory()             |
      |               |                  |----- state ------>|                  |
      |               |                  |                   |-- load(WM+RAG)  |
      |               |                  |                   |-- LLM + tools   |
      |               |                  |                   |-- save(WM)      |
      |               |                  |                   |---- state ------>|
      |               |                  |                   |                  |-- load(WM)
      |               |                  |                   |                  |-- LLM(report)
      |<----------- final state --------------------------------------------------------|
      |               |                  |                   |                  |

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.memory_management.shared_memory_multi_agent
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage

# CONNECTION: This script is Pattern 5 — it demonstrates ALL memory tiers
# working together in a single multi-agent pipeline:
#   WorkingMemory   (Tier 2) — carry intermediate state between nodes
#   LongTermMemory  (Tier 4) — retrieve medical guidelines via RAG
#   MemorySaver     (Tier 3) — persist conversation state via checkpointer
# See memory/__init__.py for the complete four-tier memory architecture diagram.
from core.config import get_llm
from core.models import PatientCase
from memory.working_memory import WorkingMemory
from tools import analyze_symptoms, assess_patient_risk, check_drug_interactions
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 5.1 — State Definition
# ============================================================

class SharedMemoryState(TypedDict):
    # Tier 1: LangGraph State
    patient_case: dict
    messages: Annotated[list, add_messages]

    # Tier 2: WorkingMemory (serialised as dict)
    working_memory: dict

    # Tier 4: ChromaDB RAG results
    guideline_context: str

    # Node outputs
    triage_assessment: str
    pharmacist_review: str
    final_report: str


# ============================================================
# STAGE 5.2 — WorkingMemory Helpers
# ============================================================

def load_working_memory(state: dict) -> WorkingMemory:
    """Deserialise WorkingMemory from state dict."""
    return WorkingMemory(initial_data=state.get("working_memory", {}))


def save_working_memory(working_mem: WorkingMemory) -> dict:
    """Serialise WorkingMemory to plain dict for state."""
    return working_mem.get_all()


# ============================================================
# STAGE 5.3 — ChromaDB Helper
# ============================================================

DRUG_GUIDELINES = [
    {
        "content": (
            "CKD KDIGO 2024: SGLT2 inhibitors recommended for CKD with eGFR >= 20. "
            "Target BP < 130/80 mmHg. Monitor K+ with ACEi/ARB + MRA. "
            "Dose-reduce ACEi if K+ > 5.0 mEq/L."
        ),
        "metadata": {"source": "KDIGO 2024", "condition": "CKD"},
    },
    {
        "content": (
            "Hyperkalemia Management: For K+ 5.0-5.5, dietary counseling and medication "
            "review. Hold/reduce MRA. Consider ACEi dose reduction. "
            "For K+ > 5.5: calcium gluconate, insulin+glucose, sodium polystyrene."
        ),
        "metadata": {"source": "ACP 2024", "condition": "Hyperkalemia"},
    },
    {
        "content": (
            "Heart Failure AHA/ACC 2024: For HFrEF, quadruple therapy: ACEi/ARB/ARNI + "
            "beta-blocker + MRA + SGLT2i. Diuretics for volume management. "
            "Monitor renal function and electrolytes."
        ),
        "metadata": {"source": "AHA/ACC 2024", "condition": "Heart Failure"},
    },
    {
        "content": (
            "Drug Interaction: Lisinopril + Spironolactone increases hyperkalemia risk, "
            "especially in CKD. Monitor K+ within 72 hours of initiation or dose change. "
            "Risk factors: eGFR < 45, age > 65, diabetes."
        ),
        "metadata": {"source": "Clinical Pharmacology", "condition": "Drug Interaction"},
    },
    {
        "content": (
            "Diabetes ADA 2024: For T2DM with CKD, prefer SGLT2i or GLP-1 RA. "
            "Metformin: hold if eGFR < 30, dose reduce if eGFR 30-45. "
            "Avoid sulfonylureas in CKD."
        ),
        "metadata": {"source": "ADA 2024", "condition": "Diabetes"},
    },
]


def _retrieve_guidelines(symptoms: list, medications: list) -> str:
    """Query ChromaDB for relevant guidelines. Returns formatted text."""
    try:
        from memory.long_term_memory import LongTermMemory

        long_term_store = LongTermMemory(collection_name="shared_memory_demo")

        documents = [guideline["content"] for guideline in DRUG_GUIDELINES]
        metadatas = [guideline["metadata"] for guideline in DRUG_GUIDELINES]
        long_term_store.add_documents(documents, metadatas)

        query = f"Patient with {', '.join(symptoms)}. Medications: {', '.join(medications)}"
        search_results = long_term_store.search(query, k=3)

        if search_results:
            lines = []
            for result_item in search_results:
                source = result_item["metadata"].get("source", "unknown")
                lines.append(f"[{source}] {result_item['content']}")
            long_term_store.clear()
            return "\n".join(lines)

        long_term_store.clear()
        return ""

    except Exception as e:
        print(f"    | [RAG] ChromaDB unavailable: {type(e).__name__}: {e}")
        return ""


# ============================================================
# STAGE 5.4 — Node Definitions
# ============================================================

def triage_node(state: SharedMemoryState) -> dict:
    """
    Triage agent — real LLM call.

    Uses LangGraph State (Tier 1) for patient data.
    Writes to WorkingMemory (Tier 2) for downstream agents.
    """
    llm = get_llm()
    tools = [analyze_symptoms, assess_patient_risk]
    agent_llm = llm.bind_tools(tools)
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a triage specialist. Use your tools to analyze the "
        "patient's symptoms and assess risk. Provide a brief triage "
        "summary (3-4 sentences). Include urgency level."
    ))
    prompt = HumanMessage(content=f"""Triage this patient:
Age: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
Vitals: {json.dumps(patient.get('vitals', {}))}
Labs: {json.dumps(patient.get('lab_results', {}))}""")

    config = build_callback_config(trace_name="shared_memory_triage")
    messages = [system, prompt]

    response = agent_llm.invoke(messages, config=config)
    while hasattr(response, "tool_calls") and response.tool_calls:
        print(f"    | [Triage] Tools: {[tc['name'] for tc in response.tool_calls]}")
        tool_node = ToolNode(tools)
        tool_results = tool_node.invoke({"messages": [response]})
        messages.extend([response] + tool_results["messages"])
        response = agent_llm.invoke(messages, config=config)

    assessment = response.content
    print(f"    | [Triage] Assessment: {len(assessment)} chars")

    # ── Write to WorkingMemory (Tier 2) ──────────────────────────────
    working_mem = load_working_memory(state)
    working_mem.set("triage_assessment", assessment[:500])
    working_mem.set("patient_symptoms", patient.get("symptoms", []))
    working_mem.set("patient_medications", patient.get("current_medications", []))
    working_mem.append_to("reasoning_trace", f"Triage: {assessment[:120]}")
    print(f"    | [Triage] Working Memory keys: {working_mem.keys()}")

    return {
        "messages": [response],
        "triage_assessment": assessment,
        "working_memory": save_working_memory(working_mem),
    }


def retrieve_guides_node(state: SharedMemoryState) -> dict:
    """
    Retrieve relevant guidelines from ChromaDB (Tier 4).

    Reads patient symptoms and medications from WorkingMemory
    to construct the RAG query.
    """
    working_mem = load_working_memory(state)
    symptoms = working_mem.get("patient_symptoms", [])
    medications = working_mem.get("patient_medications", [])

    print(f"    | [Retrieve] Querying ChromaDB for: {symptoms[:3]}...")
    retrieved_guidelines = _retrieve_guidelines(symptoms, medications)

    if retrieved_guidelines:
        print(f"    | [Retrieve] Found {retrieved_guidelines.count('[') } guideline references")
    else:
        print(f"    | [Retrieve] No guidelines retrieved")

    working_mem.append_to("reasoning_trace", f"Retrieve: {len(retrieved_guidelines)} chars of guidelines")

    return {
        "guideline_context": retrieved_guidelines,
        "working_memory": save_working_memory(working_mem),
    }


def pharmacist_node(state: SharedMemoryState) -> dict:
    """
    Pharmacist agent — reads from WorkingMemory (Tier 2)
    and ChromaDB results (Tier 4).

    Demonstrates memory tier integration: the pharmacist
    combines data from upstream agents (via WorkingMemory) with
    retrieved knowledge (via RAG).
    """
    llm = get_llm()
    tools = [check_drug_interactions]
    agent_llm = llm.bind_tools(tools)

    working_mem = load_working_memory(state)
    triage_assessment = working_mem.get("triage_assessment", "No triage available")
    medications = working_mem.get("patient_medications", [])
    retrieved_guidelines = state.get("guideline_context", "")

    print(f"    | [Pharmacist] Read from WorkingMemory: triage ({len(triage_assessment)} chars)")
    print(f"    | [Pharmacist] Read from RAG: {len(retrieved_guidelines)} chars guidelines")

    patient = state["patient_case"]
    system = SystemMessage(content=(
        "You are a clinical pharmacologist. Review the triage assessment "
        "and check drug interactions. Reference the guidelines provided. "
        "Provide a concise pharmacology review (3-4 sentences)."
    ))
    prompt = HumanMessage(content=f"""Review this case:

TRIAGE (from upstream agent via WorkingMemory):
{triage_assessment}

Patient Medications: {', '.join(medications)}
Allergies: {', '.join(patient.get('allergies', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}

{f"GUIDELINES (from ChromaDB):{chr(10)}{retrieved_guidelines}" if retrieved_guidelines else "No guideline context available."}

Check drug interactions and provide your review.""")

    config = build_callback_config(trace_name="shared_memory_pharma")
    messages = [system, prompt]

    response = agent_llm.invoke(messages, config=config)
    while hasattr(response, "tool_calls") and response.tool_calls:
        print(f"    | [Pharmacist] Tools: {[tool_call['name'] for tool_call in response.tool_calls]}")
        tool_node = ToolNode(tools)
        tool_results = tool_node.invoke({"messages": [response]})
        messages.extend([response] + tool_results["messages"])
        response = agent_llm.invoke(messages, config=config)

    pharmacist_review_text = response.content
    print(f"    | [Pharmacist] Review: {len(pharmacist_review_text)} chars")

    working_mem.set("pharmacist_review", pharmacist_review_text[:500])
    working_mem.set("medication_check_complete", True)
    working_mem.append_to("reasoning_trace", f"Pharmacist: {pharmacist_review_text[:120]}")

    return {
        "messages": [response],
        "pharmacist_review": pharmacist_review_text,
        "working_memory": save_working_memory(working_mem),
    }


def report_node(state: SharedMemoryState) -> dict:
    """
    Report generator — reads FULL WorkingMemory (Tier 2).

    Uses to_context_string() to inject all accumulated
    findings into a single LLM prompt.
    """
    llm = get_llm()
    working_mem = load_working_memory(state)

    full_context = working_mem.to_context_string(max_length=2000)
    print(f"    | [Report] Full WorkingMemory context: {len(full_context)} chars, keys: {working_mem.keys()}")

    system = SystemMessage(content=(
        "You are a clinical report writer. Compile a clinical summary "
        "from the working memory context. Structure as: "
        "1. Triage Summary 2. Pharmacology Findings 3. Recommendations. "
        "Keep under 200 words. Include 'Consult your healthcare provider'."
    ))
    prompt = HumanMessage(content=f"""Compile a clinical report from these findings:

{full_context}

Write a concise, structured clinical summary.""")

    config = build_callback_config(trace_name="shared_memory_report")
    response = llm.invoke([system, prompt], config=config)
    report = response.content

    working_mem.append_to("reasoning_trace", "Report: Clinical summary generated")
    working_mem.set("report_complete", True)

    print(f"    | [Report] Generated: {len(report)} chars")

    return {
        "messages": [response],
        "final_report": report,
        "working_memory": save_working_memory(working_mem),
    }


# ============================================================
# STAGE 5.5 — Graph Construction
# ============================================================

def build_shared_memory_graph():
    """
    Build the integration pipeline using ALL memory tiers.

    Tier 1: LangGraph State (TypedDict)
    Tier 2: WorkingMemory (dict in state)
    Tier 3: MemorySaver (checkpoints)
    Tier 4: ChromaDB (long-term RAG)
    """
    workflow = StateGraph(SharedMemoryState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("retrieve_guides", retrieve_guides_node)
    workflow.add_node("pharmacist", pharmacist_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "triage")
    workflow.add_edge("triage", "retrieve_guides")
    workflow.add_edge("retrieve_guides", "pharmacist")
    workflow.add_edge("pharmacist", "report")
    workflow.add_edge("report", END)

    # Tier 3: MemorySaver for checkpoint persistence
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# STAGE 5.6 — Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  SHARED MEMORY MULTI-AGENT")
    print("  Pattern: all memory tiers in one clinical pipeline")
    print("=" * 70)

    print("""
    Memory tiers in this pipeline:

      TIER 1 — LangGraph State (TypedDict)
        patient_case, messages, assessment, plan
        Scope: one invoke() call

      TIER 2 — WorkingMemory (scratchpad)
        triage_findings, risk_level, pharmacist_review
        Scope: one pipeline execution

      TIER 3 — MemorySaver (checkpoints)
        Full state saved after every node
        Scope: per thread_id, until deleted

      TIER 4 — ChromaDB RAG (long-term)
        Drug guidelines, clinical protocols
        Scope: permanent, shared across all runs
    """)

    patient = PatientCase(
        patient_id="PT-SM-001",
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
        "patient_case": patient.model_dump(),
        "messages": [],
        "working_memory": {},
        "guideline_context": "",
        "triage_assessment": "",
        "pharmacist_review": "",
        "final_report": "",
    }

    print(f"    Patient: {patient.patient_id} | {patient.age}y {patient.sex}")
    print(f"    Complaint: {patient.chief_complaint}")
    print(f"    History: {', '.join(patient.medical_history)}")
    print(f"    Medications: {', '.join(patient.current_medications)}")
    print()

    graph = build_shared_memory_graph()
    config = {"configurable": {"thread_id": "shared-memory-001"}}
    print("    Graph compiled with MemorySaver.\n")
    print("    " + "-" * 60)

    result = graph.invoke(initial_state, config=config)

    # ── Display results ───────────────────────────────────────────────
    print("\n    " + "=" * 60)
    print("    PIPELINE RESULTS")
    print("    " + "=" * 60)

    print(f"\n    TRIAGE ASSESSMENT:")
    for line in result["triage_assessment"][:300].split("\n"):
        if line.strip():
            print(f"      {line}")

    print(f"\n    PHARMACIST REVIEW:")
    for line in result["pharmacist_review"][:300].split("\n"):
        if line.strip():
            print(f"      {line}")

    if result.get("guideline_context"):
        print(f"\n    GUIDELINES RETRIEVED (ChromaDB):")
        for line in result["guideline_context"][:300].split("\n"):
            if line.strip():
                print(f"      {line}")

    print(f"\n    FINAL REPORT:")
    print(f"    {'─' * 50}")
    for line in result["final_report"].split("\n"):
        print(f"    | {line}")

    # ── WorkingMemory final state ─────────────────────────────────────
    working_memory_final = result.get("working_memory", {})
    print(f"\n    WORKING MEMORY — FINAL STATE:")
    print(f"    Keys: {list(working_memory_final.keys())}")
    if "reasoning_trace" in working_memory_final:
        print(f"    Reasoning trace ({len(working_memory_final['reasoning_trace'])} entries):")
        for step_index, entry in enumerate(working_memory_final["reasoning_trace"]):
            print(f"      [{step_index}] {entry[:100]}")

    # ── Checkpoint inspection (Tier 3) ────────────────────────────────
    print(f"\n    CHECKPOINT (MemorySaver):")
    snapshot = graph.get_state(config)
    saved_keys = list(snapshot.values.keys())
    print(f"    Saved state keys: {saved_keys}")
    print(f"    State is persisted — can be resumed with same thread_id")

    print("\n\n" + "=" * 70)
    print("  SHARED MEMORY MULTI-AGENT COMPLETE")
    print("=" * 70)
    print("""
    Memory tier summary:

      Tier 1 (State):      patient_case, messages, outputs
                            → one invoke(), lost after
      Tier 2 (WorkingMemory): triage_findings, pharmacist_review, trace
                            → one pipeline, inter-agent communication
      Tier 3 (Checkpoint): full state after every node
                            → persist across requests, inspectable
      Tier 4 (ChromaDB):   drug guidelines, clinical protocols
                            → permanent, shared across all users/runs

    Design principles:
      - Use the RIGHT tier for the RIGHT data
      - State for structural data (patient, messages)
      - WorkingMemory for inter-agent communication (findings, trace)
      - Checkpoints for persistence and fault recovery
      - ChromaDB for domain knowledge that evolves

    This completes the memory management pattern series:
      1. working_memory_scratchpad  — inter-agent scratchpad
      2. checkpoint_persistence     — cross-invocation state
      3. semantic_retrieval         — ChromaDB RAG
      4. conversation_memory        — multi-turn summarisation
      5. shared_memory_multi_agent  — all tiers combined
    """)


if __name__ == "__main__":
    main()
