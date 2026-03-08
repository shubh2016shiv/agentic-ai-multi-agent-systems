# PDF Knowledge Ingestion Pipeline
## Technical Design Document — Medical Guidelines RAG System

**Version:** 1.0.0  
**Author:** GenAI Platform Engineering  
**Domain:** Clinical / Medical Guidelines  
**Last Updated:** 2026-03-08  

---

## Table of Contents

1. [Objective](#1-objective)
2. [Scope & Constraints](#2-scope--constraints)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Component Design (LLD)](#4-component-design-lld)
   - 4.1 [PDF Parser Layer](#41-pdf-parser-layer)
   - 4.2 [Figure & Flowchart Processor](#42-figure--flowchart-processor)
   - 4.3 [Chunking Strategy](#43-chunking-strategy)
   - 4.4 [Embedding Layer](#44-embedding-layer)
   - 4.5 [Vector Store (ChromaDB)](#45-vector-store-chromadb)
   - 4.6 [Document Registry](#46-document-registry)
5. [Metadata Schema](#5-metadata-schema)
6. [Ingestion Flow (Step-by-Step)](#6-ingestion-flow-step-by-step)
7. [Duplicate Detection Strategy](#7-duplicate-detection-strategy)
8. [Edge Cases & Failure Handling](#8-edge-cases--failure-handling)
9. [Failure Taxonomy & Recovery Matrix](#9-failure-taxonomy--recovery-matrix)
10. [Observability & Monitoring](#10-observability--monitoring)
11. [Evaluation Strategy](#11-evaluation-strategy)
12. [Technology Decisions — Rationale](#12-technology-decisions--rationale)

---

## 1. Objective

Build a **robust, scalable, idempotent PDF ingestion pipeline** that converts complex clinical guideline PDFs — including multi-column layouts, decision flowcharts, and dense tables — into high-quality vector embeddings stored in ChromaDB, suitable for use in a Medical RAG system enriching a drug knowledge base.

### Goals

- Accurately parse and index 100s of clinical PDFs (ACC guidelines, formularies, drug monographs)
- Handle complex layouts: 2-column text, multi-page tables, clinical decision flowcharts
- Support parent-child chunking optimised for medical reasoning retrieval
- Be idempotent: re-running the pipeline on the same PDF must not create duplicates
- Be resilient: partial failures must not block other PDFs; failed chunks must be recoverable
- Produce rich, structured metadata per chunk for precise retrieval and auditability

### Non-Goals (v1.0)

- Real-time ingestion (batch pipeline only)
- Multi-language document support
- Patient-specific or PHI-containing documents
- Fine-tuning the embedding model

---

## 2. Scope & Constraints

| Dimension | Value |
|---|---|
| Input Document Types | Clinical guidelines, drug monographs, formulary PDFs |
| Volume | Up to 500 PDFs per batch run |
| Avg PDF Size | 20–60 pages, 2-column layout |
| Special Content | Flowcharts, decision trees, multi-page tables, header/footer noise |
| Vector Store | ChromaDB (self-hosted) |
| Embedding Model | BAAI/bge-m3 (primary), pubmedbert (re-rank cross-encoder) |
| Parser | Docling (primary), Mistral OCR (figure fallback) |
| Parent Chunk Store | MongoDB (parent chunks + doc registry) |
| Runtime Env | Python 3.11+, CUDA GPU for embeddings |

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PDF SOURCE LAYER                                  │
│   Local filesystem / S3 bucket / SharePoint                              │
│   e.g. hyperlipidemia.pdf, drug_monograph_warfarin.pdf ...               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   INGESTION ORCHESTRATOR                                  │
│                                                                          │
│  • Scans source directory for new / modified PDFs                        │
│  • Computes doc_id = SHA-256 of PDF bytes                                │
│  • Checks Document Registry → SKIP if already ingested                   │
│  • Dispatches each PDF to the parsing pipeline                           │
│  • Tracks per-PDF job state (pending / running / done / failed)          │
│  • Technology: Python + APScheduler or Celery                            │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
               ┌─────────────┴──────────────┐
               ▼                            ▼
┌──────────────────────┐     ┌──────────────────────────────────────────┐
│   DOCUMENT REGISTRY  │     │            PDF PARSER LAYER               │
│   (MongoDB / SQLite) │     │                                           │
│                      │     │  PRIMARY: Docling                         │
│  • doc_id hash       │     │  • 2-column layout detection              │
│  • ingestion status  │     │  • Table extraction (TableFormer model)   │
│  • chunk counts      │     │  • Section heading hierarchy              │
│  • failure logs      │     │  • Header/footer stripping                │
│  • version tracking  │     │  • Bounding box output for figures        │
└──────────────────────┘     │                                           │
                             │  FALLBACK: Mistral OCR / pdfplumber       │
                             │  • Scanned PDF handling                   │
                             │  • Table spot-validation                  │
                             └───────────────┬──────────────────────────┘
                                             │
                              ┌──────────────┴────────────────────┐
                              ▼                                    ▼
               ┌──────────────────────────┐     ┌───────────────────────────────┐
               │   TEXT / TABLE CHUNKS    │     │   FIGURE / FLOWCHART CHUNKS   │
               │                          │     │                               │
               │  Parsed sections + tables│     │  Bounding boxes → crop image  │
               │  sent to Chunking Layer  │     │  → Vision LLM captioning      │
               │                          │     │  → Structured description     │
               └──────────┬───────────────┘     └───────────────┬───────────────┘
                          │                                      │
                          └────────────────┬─────────────────────┘
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CHUNKING LAYER                                  │
│                                                                          │
│  Strategy: Parent-Child Chunking                                         │
│                                                                          │
│  PARENT CHUNKS (1200–1500 tokens)                                        │
│  → Stored in MongoDB only                                                │
│  → NOT embedded                                                          │
│  → Returned to LLM as context after child retrieval                      │
│                                                                          │
│  CHILD CHUNKS (300–400 tokens, 20% overlap)                              │
│  → Embedded + stored in ChromaDB                                         │
│  → Each carries parent_chunk_id reference                                │
│  → Chunk boundaries respect: headings, tables, figure captions           │
│                                                                          │
│  ATOMIC RULES                                                            │
│  → Tables: never split — one atomic chunk                                │
│  → Figures: caption + description = one chunk                            │
│  → Force boundary at every H1/H2 heading                                 │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DEDUPLICATION LAYER                                │
│                                                                          │
│  Layer 1: doc-level   SHA-256 hash → check Document Registry            │
│  Layer 2: chunk-level SHA-256(heading + text + page) → check ChromaDB   │
│  Layer 3: semantic    embed first 3 chunks → cosine sim > 0.97 → flag   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EMBEDDING LAYER                                   │
│                                                                          │
│  Model: BAAI/bge-m3  (8192 token context, self-hosted)                  │
│  Cross-encoder re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2           │
│  Medical precision boost: NeuML/pubmedbert-base-embeddings (re-rank)     │
│                                                                          │
│  • Batch size: 32 chunks per forward pass                                │
│  • Guard: assert len(tokens) < model_max * 0.9 before embed             │
│  • Retry: exponential backoff, 5 attempts on OOM / timeout               │
│  • Queue: Redis for chunks pending embed on failure                      │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    VECTOR STORE (ChromaDB)                               │
│                                                                          │
│  Collection: medical_guidelines_v1                                       │
│  Stored per chunk: embedding vector + full metadata                      │
│  Batched writes (50 chunks/batch) with write-ahead log                   │
│  Soft-delete support: is_superseded flag for version updates             │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                DOCUMENT REGISTRY UPDATE                                  │
│  Status: ingested | partial | failed                                     │
│  Chunk counts written, failed chunk indices logged                       │
│  Prometheus metrics emitted                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design (LLD)

### 4.1 PDF Parser Layer

#### Primary: Docling

Docling uses DocLayNet for layout analysis and TableFormer for table structure, making it the strongest open-source choice for complex clinical PDFs.

**Key Docling capabilities used:**

| Capability | How it helps |
|---|---|
| `DocLayNet` layout model | Correctly reads 2-column text in reading order |
| `TableFormer` | Converts multi-page tables to clean Markdown |
| Heading hierarchy export | Enables semantic chunk boundary enforcement |
| Bounding box output | Provides coordinates to crop figure images |
| Header/footer detection | Strips "JACC VOL. 78, NO. 9, 2021 / Virani et al." noise |

**Parser output contract:**

```python
@dataclass
class ParsedDocument:
    doc_id: str                        # SHA-256 of PDF bytes
    pdf_name: str                      # "hyperlipidemia.pdf"
    total_pages: int
    is_ocr: bool                       # True if scanned PDF
    parser_version: str                # "docling-2.26.0"
    sections: List[ParsedSection]      # ordered text sections
    tables: List[ParsedTable]          # extracted tables as markdown
    figures: List[ParsedFigure]        # bounding box + caption + page
    parse_warnings: List[str]          # non-fatal issues logged
```

**Post-parse sanitisation (always applied):**

```python
HEADER_FOOTER_PATTERNS = [
    r"JACC VOL\.\s+\d+,\s+NO\.\s+\d+,\s+\d{4}",
    r"Virani et al\.",
    r"Hypertriglyceridemia Management Expert Consensus.*",
    r"AUGUST \d+, \d{4}:\d+\s*[-–]\s*\d+",
    r"^\d{3,4}$"  # bare page numbers
]

def sanitise_text(text: str) -> str:
    for pattern in HEADER_FOOTER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    return text.strip()
```

#### Fallback: pdfplumber + Mistral OCR

| Trigger | Fallback Used |
|---|---|
| Docling parse score < 0.7 | pdfplumber for text re-extraction |
| PDF has no embedded text layer | Mistral OCR API for full-page OCR |
| Table cell count mismatch | pdfplumber table spot-check |

---

### 4.2 Figure & Flowchart Processor

This is the highest-risk component because flowcharts in clinical guidelines (like ACC Figures 3–6) encode **critical decision logic** (LDL-C thresholds, drug selection pathways, dosing algorithms). Silently skipping them creates dangerous gaps in the RAG system.

#### Pipeline

```
ParsedFigure (bbox + page + caption)
        │
        ▼
extract_page_region(pdf, bbox, page)  ← high-res crop (300 DPI)
        │
        ▼
vision_llm_describe(image, figure_type)
        │  Prompt (see below)
        ▼
FigureChunk(description, chunk_type="flowchart_description")
        │
        ▼
Attach to nearest text parent via parent_chunk_id
```

#### Vision LLM Prompt — Flowchart

```
You are a clinical content extractor processing an image from an ACC/AHA 
medical guideline.

This image is a clinical decision flowchart.

Extract and describe ALL of the following as structured prose:
1. The patient population this flowchart applies to (title/header text)
2. Every decision node: the exact condition text and numerical thresholds
   (e.g. "LDL-C < 70 mg/dL", "Fasting TG ≥ 150 mg/dL")
3. Every action box: the recommended therapy or intervention
4. All branching paths: describe each branch and what triggers it
5. Any footnotes or annotations visible in the figure
6. Drug names, drug classes, and dosing information present

Output format:
[FIGURE TYPE]: Clinical Decision Flowchart
[PATIENT POPULATION]: <text>
[DECISION PATHWAY]:
  Step 1: <condition> → <action>
  Step 2: If <condition A> → <branch A action>
          If <condition B> → <branch B action>
  ...
[DRUGS MENTIONED]: <list>
[KEY THRESHOLDS]: <list of all numerical values>
[FOOTNOTES]: <text>
```

#### Vision LLM Prompt — Table

```
You are a clinical content extractor.
This image contains a medical table from an ACC guideline.

1. Reproduce the table content as Markdown
2. After the table, write a 2-3 sentence plain-language summary 
   of what this table communicates clinically
3. List all drug names, dosages, and clinical thresholds mentioned

If the table is partially cut off, note what appears to be missing.
```

#### Fallback if Vision LLM fails

```python
figure_chunk = FigureChunk(
    content=f"[FIGURE: {figure.caption}] — Full description pending. "
            f"See page {figure.page} of {doc.pdf_name}.",
    chunk_type="figure_description_pending",
    requires_reprocessing=True
)
# This chunk is stored and flagged for nightly retry job
```

---

### 4.3 Chunking Strategy

#### Why Parent-Child for Medical PDFs

Medical guidelines have a hierarchical reasoning structure:

```
Section: "6.2.1 Adults With Clinical ASCVD"   ← PARENT
    ├── "LDL-C risk-based therapy..."           ← CHILD (retrieved)
    ├── "Very high risk patients..."            ← CHILD (retrieved)
    └── "For ASCVD not at very high risk..."    ← CHILD (retrieved)
```

The child chunk is precise enough to match a specific query. The parent chunk provides the clinical context the LLM needs to reason correctly. Returning only a child chunk in isolation risks stripping the therapeutic rationale.

#### Chunk Size Decisions

| Chunk Type | Size | Reasoning |
|---|---|---|
| Child chunk | 300–400 tokens | Precise retrieval; fits well within embedding model's optimal range |
| Overlap | 20% (~60–80 tokens) | Prevents cutting mid-sentence in clinical criteria lists |
| Parent chunk | 1200–1500 tokens | Covers a full clinical sub-section; passed to LLM as context |
| Table chunk | Atomic (no split) | Tables must be read as a unit; splitting breaks row context |
| Figure chunk | 400–600 tokens | Vision description is already summarised; keep whole |

#### Boundary Rules (enforced in code)

```python
class ChunkBoundaryEnforcer:
    
    FORCE_BREAK_ON = [
        DoclingNodeType.HEADING_1,
        DoclingNodeType.HEADING_2,
        DoclingNodeType.HEADING_3,
    ]
    
    ATOMIC_UNITS = [
        DoclingNodeType.TABLE,           # never split
        DoclingNodeType.FIGURE,          # never split
        DoclingNodeType.FIGURE_CAPTION,  # always attach to its figure
        DoclingNodeType.LIST,            # prefer to keep list items together
    ]
    
    def should_break(self, node, current_token_count) -> bool:
        if node.type in self.FORCE_BREAK_ON:
            return True
        if node.type in self.ATOMIC_UNITS:
            return False  # handled separately
        if current_token_count >= CHILD_CHUNK_MAX_TOKENS:
            return True   # soft break at sentence boundary
        return False
```

#### Two Parallel Chunk Stores

```
Parent Chunks → MongoDB collection: doc_parent_chunks
    {
        _id: parent_chunk_id,
        doc_id: doc_id,
        content: "<full 1200-1500 token text>",
        section_heading: "6.2.1 Adults With ASCVD",
        page_start: 19,
        page_end: 21,
        child_chunk_ids: ["cid_001", "cid_002", "cid_003"]
    }

Child Chunks → ChromaDB collection: medical_guidelines_v1
    Stored as: embedding vector + metadata (no raw text in vector store)
    Raw text stored in metadata field (ChromaDB supports this up to 5MB per doc)
```

---

### 4.4 Embedding Layer

#### Model Selection Rationale

```
PRIMARY EMBEDDER:  BAAI/bge-m3
  Context window: 8192 tokens
  Architecture:   Multi-granularity retrieval (dense + sparse + ColBERT)
  Self-hosted:    Yes (HuggingFace)
  Why:            Best general retrieval benchmarks; handles long clinical passages;
                  supports hybrid search out of the box

MEDICAL RE-RANKER: cross-encoder/ms-marco-MiniLM-L-6-v2 
                   + NeuML/pubmedbert-base-embeddings (for re-ranking signal)
  Why:            General tokenizers misinterpret medical abbreviations
                  (e.g. "IPE", "PCSK9i", "ASCVD"). PubMedBERT improves
                  precision for medical term matching at re-rank stage.
```

#### Retrieval Flow (at query time)

```
User Query
    │
    ▼
Embed query with bge-m3
    │
    ▼
ChromaDB ANN search → top-50 child chunks (cosine similarity)
    │
    ▼
Re-rank top-50 using cross-encoder → top-10
    │
    ▼
For each of top-10: fetch parent_chunk from MongoDB
    │
    ▼
Pass parent chunks (full context) to LLM for generation
```

#### Embedding Guard — Token Overflow Prevention

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
MODEL_MAX_TOKENS = 8192
SAFETY_MARGIN = 0.9

def safe_embed(text: str, chunk_id: str) -> List[float]:
    tokens = tokenizer.encode(text)
    if len(tokens) > MODEL_MAX_TOKENS * SAFETY_MARGIN:
        logger.warning(
            f"Chunk {chunk_id} has {len(tokens)} tokens — "
            f"exceeds safe limit. Will be truncated."
        )
        text = tokenizer.decode(tokens[:int(MODEL_MAX_TOKENS * SAFETY_MARGIN)])
    return embedder.embed(text)
```

---

### 4.5 Vector Store (ChromaDB)

#### Collection Design

```python
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="medical_guidelines_v1",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,   # higher = better index quality
        "hnsw:M": 48,                  # higher = better recall, more memory
    }
)
```

#### Batched Write with WAL

```python
BATCH_SIZE = 50

class ChromaWriter:
    def __init__(self, collection, wal_path="./wal.jsonl"):
        self.collection = collection
        self.wal = open(wal_path, "a")

    def write_batch(self, chunks: List[ChildChunk]):
        # Write to WAL first
        for chunk in chunks:
            self.wal.write(json.dumps({"chunk_id": chunk.chunk_id, 
                                        "status": "pending"}) + "\n")
        self.wal.flush()
        
        # Write to ChromaDB
        try:
            self.collection.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=[c.embedding for c in chunks],
                documents=[c.content for c in chunks],
                metadatas=[c.metadata for c in chunks]
            )
            # Mark WAL entries as committed
            for chunk in chunks:
                self.wal.write(json.dumps({"chunk_id": chunk.chunk_id, 
                                            "status": "committed"}) + "\n")
        except Exception as e:
            logger.error(f"ChromaDB write failed: {e}")
            raise
```

---

### 4.6 Document Registry

The Document Registry is the **source of truth** for the pipeline. ChromaDB is an index. The Registry tells you what has been processed, what failed, and what can be safely retried.

#### Schema (MongoDB)

```json
{
    "_id": "sha256:abc123...",
    "pdf_name": "hyperlipidemia_ACC_2021.pdf",
    "pdf_source_path": "guidelines/cardiology/",
    "file_size_bytes": 2456789,
    "status": "ingested",
    "ingestion_version": "v1.2.0",
    "parser": "docling-2.26.0",
    "embedding_model": "bge-m3",
    "total_pages": 34,
    "total_chunks_parent": 12,
    "total_chunks_child": 87,
    "chunks_embedded": 87,
    "chunks_failed": 0,
    "failed_chunk_indices": [],
    "figures_processed": 6,
    "figures_pending_description": 0,
    "ingested_at": "2026-03-08T12:00:00Z",
    "last_updated_at": "2026-03-08T12:04:33Z",
    "is_superseded": false,
    "superseded_by_doc_id": null,
    "parse_warnings": [],
    "tags": ["cardiology", "ACC", "2021", "hypertriglyceridemia"]
}
```

**Status state machine:**

```
pending ──► running ──► ingested
                │
                ├──► partial   (some chunks failed, rest succeeded)
                │      │
                │      └──► retry_pending ──► ingested | failed
                │
                └──► failed    (parser crashed, PDF unreadable)
                       │
                       └──► manual_review
```

---

## 5. Metadata Schema

Each child chunk stored in ChromaDB carries this metadata:

```json
{
    "chunk_id": "sha256:chunk_content_hash",
    "parent_chunk_id": "sha256:parent_hash",
    "doc_id": "sha256:pdf_bytes_hash",

    "pdf_name": "hyperlipidemia_ACC_2021.pdf",
    "pdf_source_path": "guidelines/cardiology/",
    "guideline_org": "ACC",
    "guideline_year": 2021,
    "therapeutic_area": "cardiology",
    "condition_focus": "hypertriglyceridemia",

    "page_numbers": [19, 20],
    "section_heading": "6.2.1 Adults With Clinical ASCVD",
    "section_depth": 3,
    "chunk_index": 42,
    "total_chunks_in_doc": 87,

    "chunk_type": "text",
    "is_table": false,
    "is_figure_description": false,
    "is_flowchart_description": false,
    "figure_description_pending": false,

    "parser_version": "docling-2.26.0",
    "embedding_model": "bge-m3-v1",
    "ingestion_pipeline_version": "v1.2.0",
    "ingested_at": "2026-03-08T12:00:00Z",

    "is_superseded": false,
    "is_ocr_sourced": false,
    "token_count": 347,
    "confidence_score": 0.97
}
```

> **Retrieval filter examples:**
> - `{"therapeutic_area": "cardiology", "is_superseded": false}` — all active cardiology chunks
> - `{"guideline_org": "ACC", "guideline_year": {"$gte": 2020}}` — recent ACC guidelines only
> - `{"chunk_type": "flowchart_description"}` — search only flowchart logic chunks

---

## 6. Ingestion Flow (Step-by-Step)

```
┌───────────────────────────────────────────────────────────────────┐
│  STEP 1: DISCOVERY                                                │
│  Scan source dir → list all PDFs → compute SHA-256 per file       │
│  Query Document Registry: is doc_id already status="ingested"?    │
│  → YES: SKIP (log as "already ingested")                          │
│  → NO:  Proceed                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 2: REGISTRATION                                             │
│  Insert doc record into Registry with status="running"            │
│  (Prevents parallel workers from processing same file)            │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 3: PARSING                                                  │
│  Run Docling on PDF                                               │
│  Post-parse sanitisation: strip headers/footers, noise            │
│  If parse_score < 0.7 → fallback to pdfplumber                   │
│  If no text layer → route to Mistral OCR                          │
│  Output: ParsedDocument (sections + tables + figures)             │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
        ┌─────────────────────┐  ┌──────────────────────────┐
        │  STEP 4A: FIGURE    │  │  STEP 4B: TEXT/TABLE      │
        │  PROCESSING         │  │  PROCESSING               │
        │                     │  │                           │
        │  For each figure:   │  │  Send to Chunking Layer   │
        │  → crop image       │  │  Apply boundary rules     │
        │  → vision LLM call  │  │  Build parent + child     │
        │  → create chunk     │  │  chunk tree               │
        │  → on fail: flag    │  │                           │
        │    as pending        │  │                           │
        └──────────┬──────────┘  └───────────┬───────────────┘
                   │                         │
                   └──────────┬──────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 5: DEDUPLICATION                                            │
│  For each child chunk:                                            │
│    chunk_id = SHA-256(section_heading + content + page_number)    │
│    Query ChromaDB: does chunk_id exist in metadata?               │
│    → EXISTS: skip this chunk (log as duplicate)                   │
│    → NEW: proceed to embedding                                    │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 6: EMBEDDING                                                │
│  Assert token_count < model_max * 0.9                             │
│  Embed with bge-m3 in batches of 32                               │
│  On failure: exponential backoff (1s, 2s, 4s, 8s, 16s)           │
│  After 5 retries: write to Redis retry queue; continue pipeline   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 7: VECTOR STORE WRITE                                       │
│  Write chunk embedding + metadata to ChromaDB in batches of 50   │
│  Write-ahead log: record pending → committed state per chunk      │
│  Store parent chunk in MongoDB                                    │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 8: REGISTRY UPDATE                                          │
│  Update doc status: ingested / partial / failed                   │
│  Write: chunks_embedded, chunks_failed, figures_pending           │
│  Emit Prometheus metrics                                          │
│  On partial: schedule retry job for failed chunk indices          │
└───────────────────────────────────────────────────────────────────┘
```

---

## 7. Duplicate Detection Strategy

Three-layer deduplication prevents redundant embeddings without blocking legitimate re-ingestion of updated documents.

### Layer 1 — Document-Level (File Hash)

```python
def check_document_duplicate(pdf_path: str, registry: DocRegistry) -> DuplicateCheckResult:
    doc_id = sha256_file(pdf_path)
    existing = registry.get(doc_id)
    
    if existing and existing["status"] == "ingested":
        return DuplicateCheckResult.SKIP
    elif existing and existing["status"] == "partial":
        return DuplicateCheckResult.RESUME  # resume from last failed chunk
    elif existing and existing["status"] == "failed":
        return DuplicateCheckResult.RETRY
    else:
        return DuplicateCheckResult.NEW
```

### Layer 2 — Chunk-Level (Content Hash)

Handles cases where a new PDF version has 80% overlapping content with an old one.

```python
def compute_chunk_id(section_heading: str, content: str, page_number: int) -> str:
    fingerprint = f"{section_heading}::{content}::{page_number}"
    return "chunk:" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

def is_chunk_duplicate(chunk_id: str, collection) -> bool:
    results = collection.get(
        where={"chunk_id": chunk_id},
        include=["metadatas"]
    )
    return len(results["ids"]) > 0
```

### Layer 3 — Semantic Near-Duplicate Detection

Catches same content with different formatting (e.g. PDF re-export changed encoding).

```python
def semantic_duplicate_check(parsed_doc: ParsedDocument, collection) -> bool:
    # Embed first 3 chunks as a fingerprint
    sample_embeddings = [embed(chunk) for chunk in parsed_doc.sections[:3]]
    
    for emb in sample_embeddings:
        results = collection.query(
            query_embeddings=[emb],
            n_results=1,
            include=["distances", "metadatas"]
        )
        if results["distances"][0][0] > 0.97:  # cosine similarity
            logger.warning(
                f"Semantic near-duplicate detected: "
                f"{parsed_doc.pdf_name} ≈ {results['metadatas'][0][0]['pdf_name']}"
            )
            return True
    return False
```

### PDF Version Update Handling

When a guideline is updated (e.g. ACC 2021 → ACC 2023):

```python
def handle_version_update(old_doc_id: str, new_doc_id: str):
    # 1. Soft-delete all old chunks
    old_chunks = collection.get(where={"doc_id": old_doc_id})
    collection.update(
        ids=old_chunks["ids"],
        metadatas=[{"is_superseded": True, "superseded_by_doc_id": new_doc_id}
                   for _ in old_chunks["ids"]]
    )
    # 2. Mark old registry entry as superseded
    registry.update(old_doc_id, {"is_superseded": True, "superseded_by": new_doc_id})
    # 3. Ingest new document normally
    ingest_document(new_doc_path)
```

---

## 8. Edge Cases & Failure Handling

### 8.1 Multi-Column Text Ordering

**Problem:** Two-column PDFs sometimes yield interleaved text (left col paragraph 1, right col paragraph 1, left col paragraph 2...) which breaks semantic coherence.

**Detection:**
```python
def validate_section_ordering(sections: List[ParsedSection]) -> bool:
    # Headings must always appear before their body content
    for i, section in enumerate(sections):
        if section.type == "body" and i > 0:
            if sections[i-1].type == "body" and sections[i-1].heading != section.heading:
                # Check if this could be column interleaving by comparing x-positions
                if sections[i].bbox.x < sections[i-1].bbox.x:
                    return False  # suspicious ordering
    return True
```

**Fix:** If ordering validation fails, re-parse using Docling's `reading_order=COLUMN_FIRST` configuration.

---

### 8.2 Tables Split Across Pages

**Problem:** Table 4 (Nutrition Recommendations, pages 15–16 of ACC PDF) spans pages. Docling's TableFormer usually handles this, but validation is needed.

```python
def validate_table_continuity(table: ParsedTable) -> bool:
    # A valid table has column headers in first row only
    # If headers appear mid-table, it was likely split and re-stitched incorrectly
    first_row = table.rows[0]
    for i, row in enumerate(table.rows[1:], 1):
        if set(row.cells) == set(first_row.cells):
            logger.warning(f"Table {table.table_id}: header row repeated at row {i} — "
                           f"possible page-split stitching error")
            return False
    return True
```

---

### 8.3 Empty Chunks After Parsing

**Problem:** Section headers with no body text, figure-only pages, merged table cells produce near-empty chunks that pollute the vector space.

```python
MIN_CHUNK_TOKENS = 50

def filter_empty_chunks(chunks: List[ChildChunk]) -> List[ChildChunk]:
    filtered = []
    for chunk in chunks:
        token_count = len(tokenizer.encode(chunk.content))
        if token_count < MIN_CHUNK_TOKENS:
            logger.debug(f"Dropping undersized chunk {chunk.chunk_id}: "
                        f"{token_count} tokens (min: {MIN_CHUNK_TOKENS})")
        else:
            filtered.append(chunk)
    return filtered
```

---

### 8.4 Cross-Reference Pollution

**Problem:** Clinical guidelines are full of `"(see Table 1)"`, `"(2)"`, `"Figure 3"` references. A child chunk containing only a reference without its target is nearly useless in retrieval.

```python
def enrich_cross_references(chunk: ChildChunk, doc: ParsedDocument) -> ChildChunk:
    # Find cross-references in chunk text
    table_refs = re.findall(r"Table\s+(\d+)", chunk.content)
    figure_refs = re.findall(r"Figure\s+(\d+)", chunk.content)
    
    for ref_num in table_refs[:2]:  # limit to 2 expansions to control token count
        if ref_num in doc.table_summaries:
            chunk.content += f"\n[Table {ref_num} Summary: {doc.table_summaries[ref_num]}]"
    
    for ref_num in figure_refs[:1]:
        if ref_num in doc.figure_summaries:
            chunk.content += f"\n[Figure {ref_num}: {doc.figure_summaries[ref_num][:200]}]"
    
    return chunk
```

---

### 8.5 OCR-Sourced PDFs

**Problem:** Legacy formulary PDFs are scanned images. OCR introduces errors in drug names, numerical thresholds (e.g. "1OO mg/dL" instead of "100 mg/dL").

```python
# Mark OCR chunks with lower confidence and apply medical term correction
def post_process_ocr_chunk(text: str) -> Tuple[str, float]:
    # Common OCR errors in medical text
    corrections = {
        r"\b1O\b": "10",    # capital O misread as zero
        r"\b0\b(?=\s+mg)": "0",  # keep as-is but flag
        r"rng/dL": "mg/dL", # common OCR error
        r"rnL": "mL",
    }
    confidence = 1.0
    for pattern, replacement in corrections.items():
        if re.search(pattern, text):
            confidence -= 0.05
            text = re.sub(pattern, replacement, text)
    
    return text, max(confidence, 0.5)
```

Chunks with `is_ocr_sourced=True` get a reduced confidence score in metadata and are down-ranked slightly during retrieval.

---

### 8.6 Token Limit Breach

**Problem:** Some parent chunks or table chunks exceed the embedding model's safe token limit. Silent truncation drops critical clinical content.

**Solution:** Enforce pre-embed token guard (covered in Section 4.4). Additionally, for tables that exceed the limit, split into row groups with a shared header row prepended to each group.

```python
def split_oversized_table(table_text: str, header_row: str, max_tokens: int) -> List[str]:
    rows = table_text.split("\n")
    chunks = []
    current_chunk = header_row + "\n"
    
    for row in rows[1:]:  # skip header
        test = current_chunk + row + "\n"
        if len(tokenizer.encode(test)) > max_tokens * 0.85:
            chunks.append(current_chunk)
            current_chunk = header_row + "\n" + row + "\n"  # restart with header
        else:
            current_chunk = test
    
    if current_chunk.strip() != header_row.strip():
        chunks.append(current_chunk)
    
    return chunks
```

---

## 9. Failure Taxonomy & Recovery Matrix

| Failure Type | When It Happens | Detection | Recovery Action | Blocks Pipeline? |
|---|---|---|---|---|
| **PDF corrupted / unreadable** | Step 3: Parser | Exception on open | Move to `/failed_queue/`, log, alert | ❌ No |
| **Password-protected PDF** | Step 3: Parser | `PasswordRequiredError` | Move to `/needs_password/`, notify team | ❌ No |
| **OCR-only / no text layer** | Step 3: Parser | text extraction returns empty | Route to Mistral OCR | ❌ No |
| **Docling parse score < 0.7** | Step 3: Parser | Low confidence score | Fallback to pdfplumber | ❌ No |
| **Multi-column ordering error** | Step 3: Post-parse | Ordering validation fails | Re-parse with `COLUMN_FIRST` mode | ❌ No |
| **Table split/stitch error** | Step 3: Post-parse | Header row repeated mid-table | Flag table, use raw text fallback | ❌ No |
| **Vision LLM timeout / quota** | Step 4A: Figure proc | HTTPError / timeout | Store as `figure_pending`, continue | ❌ No |
| **Chunk below min token count** | Step 5: Dedup | Token count check | Drop chunk, log warning | ❌ No |
| **Chunk exceeds model token limit** | Step 6: Embed | Token count assertion | Truncate with warning | ❌ No |
| **Embedding model OOM** | Step 6: Embed | CUDA OOM exception | Exponential backoff × 5, then Redis queue | ❌ No |
| **Embedding model down** | Step 6: Embed | Connection refused | Queue chunks to Redis, process when up | ❌ No |
| **ChromaDB write failure** | Step 7: Write | Exception on `.add()` | WAL replay on restart | ❌ No |
| **ChromaDB disk full** | Step 7: Write | IOError | Pause pipeline, alert ops | ⚠️ Yes (pause) |
| **MongoDB connection lost** | Step 7/8 | ConnectionError | Retry × 3, then fail the doc | ❌ No |
| **Partial ingestion (N/M chunks)** | Step 8 | Registry: `chunks_failed > 0` | Status = `partial`, schedule retry | ❌ No |
| **Same PDF ingested twice** | Step 1 | Registry hash check | `SKIP` — idempotent | ❌ No |
| **Updated PDF (same name)** | Step 1 | Hash differs, name same | New doc_id, soft-delete old chunks | ❌ No |

### Retry Architecture

```
Redis Queue: ingestion_retry_queue
    ├── chunk_retry      → individual failed chunks
    ├── figure_retry     → pending figure descriptions
    └── doc_retry        → fully failed documents

Nightly Retry Job (cron: 02:00 UTC):
    1. Pull all items from retry queues
    2. Re-attempt embedding / figure description / parse
    3. Update registry status
    4. Alert if retry_count > 3 → escalate to manual review
```

---

## 10. Observability & Monitoring

### Prometheus Metrics

```python
# Emitted at end of each document ingestion
ingestion_docs_total{status="ingested|partial|failed|skipped"} 
ingestion_chunks_total{type="text|table|figure|flowchart"}
ingestion_chunks_embedded_total
ingestion_chunks_failed_total
ingestion_duration_seconds{pdf_name=...}
ingestion_figures_pending_total
embedding_queue_depth          # Redis retry queue size
chroma_collection_size         # total vectors in ChromaDB
```

### Alerting Thresholds

| Metric | Warning | Critical |
|---|---|---|
| `ingestion_chunks_failed_total` (per doc) | > 5% | > 20% |
| `ingestion_figures_pending_total` | > 10 | > 50 |
| `embedding_queue_depth` | > 100 | > 500 |
| `ingestion_duration_seconds` | > 5 min/doc | > 15 min/doc |
| Parse failures in 24h window | > 5 | > 20 |

---

## 11. Evaluation Strategy

### Golden Evaluation Set

Build a set of 30–50 question-answer pairs derived from your target PDFs **before batch ingestion begins**. This is your ground truth.

Example Q&A pairs from the ACC Hypertriglyceridemia PDF:

```
Q: What is the recommended first-line treatment for a patient with 
   clinical ASCVD and LDL-C < 70 mg/dL and fasting TG between 
   150-499 mg/dL?
A: Icosapent ethyl (IPE) should be considered. (Section 6.2.1, Figure 3)

Q: At what fasting triglyceride level does persistent hypertriglyceridemia 
   trigger consideration of nonstatin therapies?
A: ≥ 150 mg/dL fasting, or ≥ 175 mg/dL nonfasting. (Section 4, Definition 1)

Q: What lifestyle intervention produces the greatest triglyceride reduction?
A: Dietary modifications (including alcohol restriction) with >70% 
   potential reduction. (Table 3)
```

### Metrics to Track

| Metric | Formula | Target |
|---|---|---|
| **Retrieval Recall@5** | % of golden answers in top-5 retrieved chunks | > 85% |
| **Retrieval Precision@5** | Relevance of top-5 chunks | > 70% |
| **Figure Coverage** | % of flowchart decisions retrievable | > 90% |
| **Duplicate Rate** | Duplicate chunk_ids / total chunks | 0% |
| **Chunk Completeness** | Chunks with token_count > 50 / total | > 99% |

Run evaluation after every batch ingestion:
```
python evaluate_retrieval.py --golden_set golden_qa.json \
                              --collection medical_guidelines_v1 \
                              --output report.json
```

---

## 12. Technology Decisions — Rationale

| Component | Choice | Rejected Alternatives | Reason |
|---|---|---|---|
| **PDF Parser** | Docling | LlamaParse, Unstructured, PyMuPDF | Docling: best open-source table extraction (TableFormer), layout understanding (DocLayNet), no per-page pricing. LlamaParse: expensive at scale. Unstructured: quality degraded recently. |
| **Figure Processing** | Mistral OCR / Claude Vision | Skip figures | Flowcharts encode critical clinical decision logic — skipping them creates dangerous RAG gaps |
| **Chunking** | Parent-Child | Fixed-size, Sentence-window | Fixed-size breaks clinical reasoning chains. Parent-child gives precision (child) + context (parent). Medical guidelines have natural parent-child section hierarchy. |
| **Embedding** | BAAI/bge-m3 | pplx-embed-context-v1-4b, OpenAI text-embedding-3 | bge-m3: self-hosted, 8192 context, best open retrieval benchmarks, supports hybrid dense+sparse search. pplx: API cost at scale. |
| **Medical Re-ranker** | NeuML/pubmedbert | None | General tokenizers misrepresent clinical abbreviations (PCSK9i, IPE, ASCVD). PubMedBERT improves re-rank precision for medical terms. |
| **Parent Chunk Store** | MongoDB | ChromaDB metadata only | ChromaDB metadata has size limits. Parent chunks (1500 tokens) need reliable structured storage with query support. MongoDB also doubles as the doc registry. |
| **Deduplication** | 3-layer (hash + content hash + semantic) | Hash only | Hash alone misses same-content PDFs with different byte layouts. Semantic layer catches near-duplicates from PDF re-exports. |
| **Write Durability** | WAL + batched writes | Direct writes | ChromaDB has no transaction rollback. WAL enables recovery of partial batch writes on crash. |

---

*End of Document*
