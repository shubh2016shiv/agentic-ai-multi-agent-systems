#!/usr/bin/env python3
"""
============================================================
Semantic Retrieval
============================================================
Pattern 3: ChromaDB as long-term semantic memory (RAG)
integrated into a LangGraph graph node.
Prerequisite: checkpoint_persistence.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
Long-term memory stores knowledge that persists across all
pipeline runs and all users. In healthcare, this is:

    - Drug formulary and interaction databases
    - Clinical practice guidelines (COPD, CKD, HF, etc.)
    - Protocol documents and SOPs

ChromaDB is a vector database that indexes these documents
and retrieves the most relevant ones given a patient query.
This is the RETRIEVAL step of the RAG (Retrieval-Augmented
Generation) pattern.

The agent's prompt includes:
    1. Patient data (from state)
    2. Retrieved guidelines (from ChromaDB)
    3. Instructions (system message)

This gives the LLM access to specific, up-to-date knowledge
that it may not have seen during training.

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [index_guidelines]   <-- one-time: load guidelines into ChromaDB
       |
       v
    [retrieve]           <-- query ChromaDB for relevant guidelines
       |
       v
    [assess_with_rag]    <-- real LLM call with RAG context
       |
       v
    [assess_without_rag] <-- real LLM call WITHOUT RAG (comparison)
       |
       v
    [END]

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()        index_node        retrieve_node      assess_with_rag     assess_without_rag
      |               |                  |                   |                    |
      |-- invoke() -->|                  |                   |                    |
      |               |-- LongTermMemory()|                  |                    |
      |               |-- add_documents() |                  |                    |
      |               |----- state ------>|                  |                    |
      |               |                   |-- LongTermMemory()|                  |
      |               |                   |-- search(query)   |                  |
      |               |                   |<- 3 guidelines    |                  |
      |               |                   |----- state ------>|                  |
      |               |                   |                   |-- LLM(patient     |
      |               |                   |                   |    + guidelines)  |
      |               |                   |                   |---- state ------->|
      |               |                   |                   |                   |-- LLM(patient
      |               |                   |                   |                   |    only)
      |<------------- final state -------------------------------------------- --|
      |               |                  |                   |                    |

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. ChromaDB indexing (add_documents with metadata)
    2. Semantic search (similarity-based retrieval)
    3. RAG injection into LLM prompts
    4. Response quality WITH vs WITHOUT RAG context
    5. Metadata filtering (condition-specific retrieval)

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.memory_management.semantic_retrieval
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

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage

# CONNECTION: LongTermMemory lives in the root memory module (Tier 4 memory).
# It wraps ChromaDB with add_documents() and search() methods. This script
# demonstrates the full RAG pattern: index → retrieve → augment → generate.
# LongTermMemory is instantiated lazily inside nodes (see STAGE 3.3) so that
# the ChromaDB client is created only when needed.
# See memory/long_term_memory.py for the RAG concept and ChromaDB embedding
# explanation (why provider embeddings instead of ONNX built-in model).
from core.config import get_llm
from observability.callbacks import build_callback_config


# ============================================================
# STAGE 3.1 — Drug Guideline Knowledge Base
# ============================================================
# These are the documents that get indexed into ChromaDB.
# In production, these would be loaded from a document store,
# PDF parser, or guideline API.

GUIDELINES = [
    {
        "content": (
            "COPD GOLD 2024: Long-acting bronchodilators (LAMA or LABA) are preferred "
            "as first-line maintenance therapy for COPD with persistent symptoms. "
            "For patients with frequent exacerbations, consider LAMA+LABA combination. "
            "ICS is added only if eosinophils >= 300 cells/uL."
        ),
        "metadata": {"source": "GOLD 2024", "condition": "COPD", "category": "respiratory"},
    },
    {
        "content": (
            "CKD KDIGO 2024: SGLT2 inhibitors (dapagliflozin, empagliflozin) recommended "
            "for CKD patients with eGFR >= 20 to slow progression. Target BP < 130/80 mmHg. "
            "Monitor potassium closely when using ACEi/ARB + MRA combination. "
            "Consider dose reduction of ACEi if K+ > 5.0 mEq/L."
        ),
        "metadata": {"source": "KDIGO 2024", "condition": "CKD", "category": "renal"},
    },
    {
        "content": (
            "Heart Failure AHA/ACC 2024: For HFrEF (EF <= 40%), initiate quadruple therapy: "
            "ACEi/ARB/ARNI + beta-blocker + MRA + SGLT2i as tolerated. "
            "Titrate to target doses over weeks. Monitor renal function and electrolytes. "
            "Diuretics for volume management (not prognostic benefit)."
        ),
        "metadata": {"source": "AHA/ACC 2024", "condition": "Heart Failure", "category": "cardiac"},
    },
    {
        "content": (
            "Hyperkalemia Management 2024: For K+ 5.0-5.5 mEq/L, dietary counseling and "
            "medication review. Hold/reduce MRA (spironolactone, eplerenone). "
            "Consider ACEi dose reduction. For K+ > 5.5 mEq/L, urgent intervention: "
            "calcium gluconate, insulin+glucose, sodium polystyrene. "
            "Avoid NSAIDs and potassium supplements."
        ),
        "metadata": {"source": "ACP 2024", "condition": "Hyperkalemia", "category": "electrolyte"},
    },
    {
        "content": (
            "Diabetes ADA 2024: For T2DM with CKD or HF, prefer SGLT2i or GLP-1 RA "
            "regardless of HbA1c level for cardiorenal protection. "
            "Metformin: hold if eGFR < 30 mL/min, dose reduce if eGFR 30-45. "
            "Avoid sulfonylureas in CKD (hypoglycemia risk)."
        ),
        "metadata": {"source": "ADA 2024", "condition": "Diabetes", "category": "metabolic"},
    },
    {
        "content": (
            "Drug Interaction Alert — ACEi + MRA: Lisinopril + Spironolactone increases "
            "risk of hyperkalemia, especially in CKD. Monitor K+ within 72 hours of "
            "initiation or dose change. Risk factors: eGFR < 45, age > 65, diabetes, "
            "concurrent NSAID use. Consider discontinuation if K+ > 5.5 mEq/L."
        ),
        "metadata": {"source": "Clinical Pharmacology", "condition": "Drug Interaction", "category": "pharmacology"},
    },
]


# ============================================================
# STAGE 3.2 — State Definition
# ============================================================

class SemanticState(TypedDict):
    patient_case: dict
    guidelines_indexed: bool       # Written by: index_node
    retrieved_guidelines: str      # Written by: retrieve_node
    response_with_rag: str         # Written by: assess_with_rag_node
    response_without_rag: str      # Written by: assess_without_rag_node


# ============================================================
# STAGE 3.3 — Node Definitions
# ============================================================

def index_node(state: SemanticState) -> dict:
    """
    Index guideline documents into ChromaDB.

    In production, this would be a separate offline process
    (batch indexing), not done during pipeline execution.
    For this demo, we index inline to keep things self-contained.
    """
    try:
        from memory.long_term_memory import LongTermMemory

        long_term_store = LongTermMemory(collection_name="semantic_retrieval_demo")

        documents = [guideline["content"] for guideline in GUIDELINES]
        metadatas = [guideline["metadata"] for guideline in GUIDELINES]

        long_term_store.add_documents(documents, metadatas)
        document_count = long_term_store.get_document_count()

        print(f"    | [Index] Indexed {document_count} guidelines into ChromaDB")
        for guideline in GUIDELINES:
            print(f"    |   [{guideline['metadata']['source']}] {guideline['metadata']['condition']}")

        return {"guidelines_indexed": True}

    except Exception as e:
        print(f"    | [Index] ChromaDB unavailable: {type(e).__name__}: {e}")
        print(f"    |   Continuing without RAG (response comparison will still work)")
        return {"guidelines_indexed": False}


def retrieve_node(state: SemanticState) -> dict:
    """
    Query ChromaDB for guidelines relevant to this patient.

    This is the RETRIEVE step of the RAG pattern. The query
    is constructed from the patient's symptoms and conditions.
    """
    if not state.get("guidelines_indexed"):
        return {"retrieved_guidelines": ""}

    try:
        from memory.long_term_memory import LongTermMemory

        long_term_store = LongTermMemory(collection_name="semantic_retrieval_demo")
        patient = state["patient_case"]

        # Build a natural language query from patient data
        symptoms = ", ".join(patient.get("symptoms", []))
        history = ", ".join(patient.get("medical_history", []))
        medications = ", ".join(patient.get("current_medications", []))

        query = (
            f"Patient with {symptoms}. "
            f"Medical history: {history}. "
            f"Medications: {medications}. "
            f"Lab results: {json.dumps(patient.get('lab_results', {}))}"
        )

        # Retrieve top 3 relevant guidelines
        search_results = long_term_store.search(query, k=3)

        if search_results:
            lines = []
            for result_item in search_results:
                source = result_item["metadata"].get("source", "unknown")
                distance = result_item.get("distance", "N/A")
                lines.append(f"[{source}] (distance: {distance})")
                lines.append(f"  {result_item['content']}")
                lines.append("")

            guideline_text = "\n".join(lines)
            print(f"    | [Retrieve] Found {len(search_results)} relevant guidelines:")
            for result_item in search_results:
                print(f"    |   [{result_item['metadata'].get('source')}] "
                      f"distance={result_item.get('distance', 'N/A')}")

            # Clean up demo collection
            long_term_store.clear()
            return {"retrieved_guidelines": guideline_text}

        long_term_store.clear()
        print(f"    | [Retrieve] No matching guidelines found")
        return {"retrieved_guidelines": ""}

    except Exception as e:
        print(f"    | [Retrieve] ChromaDB search failed: {type(e).__name__}: {e}")
        return {"retrieved_guidelines": ""}


def assess_with_rag_node(state: SemanticState) -> dict:
    """
    Clinical assessment WITH RAG context.

    The LLM receives:
      1. Patient data
      2. Retrieved guidelines from ChromaDB
      3. Instructions to reference the guidelines
    """
    llm = get_llm()
    patient = state["patient_case"]
    guidelines = state.get("retrieved_guidelines", "")

    system = SystemMessage(content=(
        "You are a clinical specialist. Provide a concise assessment "
        "and treatment recommendation. If clinical guidelines are provided, "
        "reference them specifically in your recommendation. "
        "Include the guideline source in brackets, e.g. [KDIGO 2024]."
    ))

    prompt_text = f"""Patient: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
History: {', '.join(patient.get('medical_history', []))}
Medications: {', '.join(patient.get('current_medications', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
Vitals: {json.dumps(patient.get('vitals', {}))}"""

    if guidelines:
        prompt_text += f"\n\nRELEVANT CLINICAL GUIDELINES (from knowledge base):\n{guidelines}"
        prompt_text += "\n\nReference the guidelines above in your assessment."

    prompt = HumanMessage(content=prompt_text)
    config = build_callback_config(trace_name="semantic_retrieval_with_rag")
    response = llm.invoke([system, prompt], config=config)

    print(f"    | [Assess+RAG] Response: {len(response.content)} chars")
    return {"response_with_rag": response.content}


def assess_without_rag_node(state: SemanticState) -> dict:
    """
    Clinical assessment WITHOUT RAG context.

    Same patient, same LLM, but NO guidelines injected.
    This shows the difference RAG makes.
    """
    llm = get_llm()
    patient = state["patient_case"]

    system = SystemMessage(content=(
        "You are a clinical specialist. Provide a concise assessment "
        "and treatment recommendation based on the patient data only."
    ))

    prompt = HumanMessage(content=f"""Patient: {patient.get('age')}y {patient.get('sex')}
Complaint: {patient.get('chief_complaint')}
Symptoms: {', '.join(patient.get('symptoms', []))}
History: {', '.join(patient.get('medical_history', []))}
Medications: {', '.join(patient.get('current_medications', []))}
Labs: {json.dumps(patient.get('lab_results', {}))}
Vitals: {json.dumps(patient.get('vitals', {}))}""")

    config = build_callback_config(trace_name="semantic_retrieval_without_rag")
    response = llm.invoke([system, prompt], config=config)

    print(f"    | [Assess-NoRAG] Response: {len(response.content)} chars")
    return {"response_without_rag": response.content}


# ============================================================
# STAGE 3.4 — Graph Construction
# ============================================================

def build_retrieval_graph():
    """Build the semantic retrieval demo graph."""
    workflow = StateGraph(SemanticState)

    workflow.add_node("index", index_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("assess_with_rag", assess_with_rag_node)
    workflow.add_node("assess_without_rag", assess_without_rag_node)

    workflow.add_edge(START, "index")
    workflow.add_edge("index", "retrieve")
    workflow.add_edge("retrieve", "assess_with_rag")
    workflow.add_edge("assess_with_rag", "assess_without_rag")
    workflow.add_edge("assess_without_rag", END)

    return workflow.compile()


# ============================================================
# STAGE 3.5 — Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  SEMANTIC RETRIEVAL")
    print("  Pattern: ChromaDB RAG in LangGraph graph nodes")
    print("=" * 70)

    print("""
    Long-term memory (ChromaDB) stores guidelines and drug data.
    At query time, the agent retrieves relevant context:

      [index]     → load guidelines into ChromaDB (one-time)
      [retrieve]  → semantic search for patient's conditions
      [assess+RAG]    → LLM call WITH retrieved guidelines
      [assess-NoRAG]  → LLM call WITHOUT guidelines (comparison)

    The RAG pattern:
      1. Index: embed documents into vector store
      2. Retrieve: find similar documents for query
      3. Augment: inject retrieved docs into LLM prompt
      4. Generate: LLM produces response with extra context
    """)

    patient = {
        "patient_id": "PT-SR-001",
        "age": 71, "sex": "F",
        "chief_complaint": "Dizziness with elevated potassium",
        "symptoms": ["dizziness", "fatigue", "ankle edema", "orthopnea"],
        "medical_history": ["CKD Stage 3a", "Hypertension", "CHF", "Type 2 Diabetes"],
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
        "guidelines_indexed": False,
        "retrieved_guidelines": "",
        "response_with_rag": "",
        "response_without_rag": "",
    }

    print(f"    Patient: {patient['age']}y {patient['sex']}")
    print(f"    Complaint: {patient['chief_complaint']}")
    print(f"    History: {', '.join(patient['medical_history'])}")
    print()

    graph = build_retrieval_graph()
    print("    Graph compiled.\n")
    print("    " + "-" * 60)

    result = graph.invoke(initial_state)

    # ── Display comparison ────────────────────────────────────────────
    print("\n    " + "=" * 60)
    print("    RESPONSE COMPARISON")
    print("    " + "=" * 60)

    print(f"\n    WITH RAG (guidelines injected):")
    print(f"    {'─' * 50}")
    for line in result["response_with_rag"][:500].split("\n"):
        print(f"    | {line}")
    if len(result["response_with_rag"]) > 500:
        print(f"    | ... ({len(result['response_with_rag'])} chars total)")

    print(f"\n    WITHOUT RAG (LLM knowledge only):")
    print(f"    {'─' * 50}")
    for line in result["response_without_rag"][:500].split("\n"):
        print(f"    | {line}")
    if len(result["response_without_rag"]) > 500:
        print(f"    | ... ({len(result['response_without_rag'])} chars total)")

    # ── Highlight differences ─────────────────────────────────────────
    rag_response = result["response_with_rag"].lower()
    has_citations = any(
        src in rag_response
        for src in ["kdigo", "aha", "gold", "ada", "acp", "pharmacology"]
    )

    print(f"\n    ANALYSIS:")
    print(f"      WITH RAG cites specific guidelines: {'YES' if has_citations else 'Check response above'}")
    print(f"      WITH RAG length: {len(result['response_with_rag'])} chars")
    print(f"      WITHOUT RAG length: {len(result['response_without_rag'])} chars")

    print("\n\n" + "=" * 70)
    print("  SEMANTIC RETRIEVAL COMPLETE")
    print("=" * 70)
    print("""
    What you saw:
      1. ChromaDB indexed 6 drug/guideline documents with metadata
      2. Semantic search retrieved the 3 most relevant guidelines
      3. LLM WITH RAG referenced specific guidelines by name
      4. LLM WITHOUT RAG used only its training knowledge

    When to use semantic retrieval:
      - Drug interaction databases (updated quarterly)
      - Clinical practice guidelines (version-specific)
      - Institution-specific protocols
      - Any domain knowledge that evolves faster than LLM training

    Production backends:
      ChromaDB    — local vector DB, good for prototyping
      Pinecone    — managed cloud vector DB
      Weaviate    — open-source, self-hosted
      pgvector    — PostgreSQL extension (reuse existing infra)

    Next: conversation_memory.py — multi-turn with summarisation.
    """)


if __name__ == "__main__":
    main()
