# Chapter 6 — Putting It All Together

> **Learning chapter** — The recommended golden path for integrating all four tiers of Memory into a production LangGraph pipeline.

---

## 6.1 The Master Integration Pattern

The most robust way to implement Memory is to use `WorkingMemory` dynamically inside nodes, `ConversationMemory` on the graph edges, and `LongTermMemory` as a tool or system prompt injector.

### 1. Define the LangGraph State
```python
from langgraph.graph import add_messages

class ClinicalState(TypedDict):
    # Tier 1 - LangGraph dict
    patient_id: str
    
    # Tier 2 - Working Memory (Serialized String)
    working_context: str
    
    # Tier 3 - Conversation Memory
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
```

### 2. The Check-in Node (Long-Term & Working Memory)
```python
from memory.long_term_memory import LongTermMemory
from memory.working_memory import WorkingMemory

ltm = LongTermMemory(collection_name="clinical_guidelines")

def triage_node(state: ClinicalState):
    # 1. Grab patient condition
    condition = state["messages"][-1].content
    
    # 2. Retrieve guidelines (Tier 4)
    guidelines = ltm.search(condition, k=1)
    
    # 3. Store findings in Working Memory (Tier 2)
    wm = WorkingMemory()
    wm.append_to("triage_notes", f"Guideline: {guidelines[0]['content']}")
    
    return {"working_context": wm.to_context_string()}
```

### 3. The Conversation Node (Rolling Summarization)
```python
from memory.conversation_memory import ConversationMemory
from langchain_openai import ChatOpenAI

cm = ConversationMemory(summarise_after=6, history_window=4)
llm = ChatOpenAI()

def summarize_node(state: ClinicalState):
    # Compress history (Tier 3)
    new_msgs, new_summary = cm.maybe_summarise(
        messages=state["messages"],
        old_summary=state.get("summary", ""),
        llm=llm
    )
    
    # NOTE: You MUST return a RemoveMessage list if you are modifying 
    # history in LangGraph. For simplicity in this pseudo-code, we 
    # pretend `new_msgs` handles the overwriting.
    return {"messages": new_msgs, "summary": new_summary}
```

### 4. Compiling the Graph (The Checkpointer)
```python
from langgraph.graph import StateGraph, START, END
from memory.checkpoint_helpers import build_checkpointed_graph

builder = StateGraph(ClinicalState)
builder.add_node("triage", triage_node)
builder.add_node("summarize", summarize_node)

builder.add_edge(START, "triage")
builder.add_edge("triage", "summarize")
builder.add_edge("summarize", END)

# Attach the Checkpointer! 
graph = build_checkpointed_graph(builder, checkpointer_type="sqlite")
```

### 5. Running the Application
```python
# The anchor for episodic memory: the Thread ID
config = {"configurable": {"thread_id": "patient_001"}}
graph.invoke({"messages": [HumanMessage("I have a cough")]}, config)
```

---

## 6.2 Scaling Considerations

**The SQLite File:** 
If you use `SqliteSaver` in production on a server running 10,000 threads, your `checkpoints.db` file will lock violently and corrupt. It is single-writer only. For enterprise environments, you MUST switch to `PostgresSaver`.

**ChromaDB Sizing:**
Chroma DB is remarkably fast, but it writes entirely to local disk. If you index 1,000,000 medical documents, `Chroma()` will take minutes to load into RAM upon server boot. Eventually, you will need to swap it out for a dedicated vector database microservice (Pinecone, Qdrant).

**Context Window Costs:**
If you set your `history_window` to 50 instead of 4, you are paying for every prompt passed to OpenAI on every turn. A 50-turn window multiplied by 1,000 users talking back and forth 10 times a day will result in a massive OpenAI bill. Be incredibly aggressive about summarizing old conversation history.
