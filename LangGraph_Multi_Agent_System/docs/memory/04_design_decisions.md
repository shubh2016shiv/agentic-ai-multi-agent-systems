# Chapter 4 — Design Decisions

> **Learning chapter** — an examination of the non-obvious choices made when designing the `memory` package, and the trade-offs involved.

---

## 4.1 Why count messages instead of tokens?

In `conversation_memory.py`:
```python
def should_summarise(self, messages: list) -> bool:
    return len(messages) > self.summarise_after
```

*The reality:* A massive, 5,000-word patient history copied into a single message takes up wildly more tokens than a 2-word answer like "Yes, doctor."

*Our Design Decision:* **Message counting is the exact right starting point.**

For educational and prototype purposes, message counting is easy to understand, requires zero external libraries, and executes instantly.

If we counted exact tokens, we would have to install `tiktoken` (which handles OpenAI's BPE encoding). However, we are running diverse LLMs (Gemini, Llama) which use completely different tokenization schemes. A 5,000 token limit in `tiktoken` might actually be 5,500 tokens to a local Llama model—crashing the system anyway. 

By simply triggering a hard compression after 6 conversational "turns," we guarantee the window stays small regardless of the underlying token math, minimizing system dependencies.

---

## 4.2 Why inject provider embeddings instead of using ChromaDB's default?

In `long_term_memory.py`, we go out of our way to retrieve embeddings via `get_embeddings()` instead of just instantiating `Chroma()`.

*The default ChromaDB approach:*
```python
vectorstore = Chroma(collection_name="docs") 
```
If you run this code, ChromaDB automatically initiates a 79MB download of the `all-MiniLM-L6-v2` ONNX model. It will then run this model literally on your CPU. 

*Our Design Decision:* **Never run unknown models locally without intent.**

1. CPU embeddings on Windows machines without configured PyTorch/CUDA environments are brutally slow.
2. It fragments the architecture. You would have your Chat completions running through the blazing fast OpenAI API, but your embeddings grinding locally on your laptop fan.
3. By explicitly passing the `.env`-configured embedding provider, we guarantee uniformity. If a developer switches the `.env` from OpenAI to Gemini, both the `ChatModel` *and* the `EmbeddingsProvider` switch instantly.

---

## 4.3 Why not put `WorkingMemory` methods directly inside the Graph State?

*The native LangGraph approach:*
```python
def __add__(a, b): return a + b

class State(TypedDict):
    findings: Annotated[list[str], __add__]
```

*Our Design Decision:* **Separation of concerns.**

Using `Annotated` reducers in LangGraph is powerful, but it makes the `State` class confusing for beginners. Every time a node returns `{"findings": ["Fever"]}`, LangGraph magically merges it.

However, sometimes an agent needs to accumulate data (`append`), while another agent needs to overwrite data (`set`), and another needs private data (`scratchpad`). Trying to shoehorn all of these mutating behaviors into custom `Annotated` reducers makes the State definition 100 lines long and unreadable.

Our `WorkingMemory` class provides a clean, zero-magic object-oriented API that explicitly handles `get`, `set`, and `append_to` while leaving the Graph State alone. 

---

## 4.4 Why is the Checkpointer abstracted via `checkpoint_helpers.py`?

*The problem:* In development, you want rapid iteration. You hit run, test the agent, and the script exits. 
In production, you need persistent states. If the server restarts, the conversation must survive.

*Our Design Decision:* **Zero-code checkpointer swapping.**

If you hardcode `PostgresSaver` into your main graph file, your developers cannot run the unit tests on planes or coffee shops without setting up a local Docker database. 

By wrapping it entirely inside `build_checkpointed_graph()`, the main developers write graph code oblivious to where the data is actually going. A single environment flag can flip the entire system from RAM-only `MemorySaver` to a distributed `PostgresSaver`.
