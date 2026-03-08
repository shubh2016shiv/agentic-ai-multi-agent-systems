#!/usr/bin/env python3
"""
============================================================
Conversation Memory
============================================================
Pattern 4: Episodic/conversation memory — how agents maintain
multi-turn conversation history and use summarisation to
control token count.
Prerequisite: semantic_retrieval.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
In a multi-turn interaction, the agent needs to remember
what was said in previous turns. This is conversation memory
(episodic memory in cognitive science terms).

The challenge: conversation history grows with every turn.
Eventually, it exceeds the LLM's context window or becomes
too expensive (tokens cost money).

Solution: rolling summarisation.

    Turn 1: [msg1]                     → send all
    Turn 2: [msg1, msg2]               → send all
    Turn 3: [msg1, msg2, msg3]         → send all
    Turn 4: [msg1..3 are old]          → summarise + [msg4]
    Turn 5: [summary, msg4, msg5]      → send summary + recent

This keeps the context window bounded while preserving
the essential information from earlier turns.

------------------------------------------------------------
GRAPH TOPOLOGY
------------------------------------------------------------

    [START]
       |
       v
    [check_history]     <-- count messages, decide if summarisation needed
       |
    route()
       |
    +--+----+
    |       |
    | full  | summarise
    v       v
    [agent] [summarise] -> [agent]
       |
       v
    [END]

    Each invoke() adds a user message, runs the agent,
    and checks if history needs summarisation.

------------------------------------------------------------
SEQUENCE DIAGRAM
------------------------------------------------------------

    main()          check_history    summarise_node      agent_node
      |                  |                |                  |
      |--- invoke(msg) ->|                |                  |
      |                  |-- count msgs   |                  |
      |                  |-- route()      |                  |
      |                  |               (if > threshold)    |
      |                  |--- "summarise" -->|                |
      |                  |               |-- compress old msgs|
      |                  |               |-- LLM(summarise)  |
      |                  |               |-- replace msgs    |
      |                  |               |---- state ------->|
      |                  |               |                   |-- LLM(respond)
      |                  |-- "agent" --->|                   |
      |                  |               (if <= threshold)   |
      |                  |-------------------------->|       |
      |                  |                           |-- LLM |
      |<------------ final state ---------------------- -----|                  
      |                  |                |                  |

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. add_messages reducer for message accumulation
    2. MemorySaver for multi-turn persistence
    3. Summarisation to control context window
    4. Message windowing (keep N recent + summary)
    5. Token-aware history management

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.memory_management.conversation_memory
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
from typing import TypedDict, Literal, Annotated

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Project imports ─────────────────────────────────────────────────────────
from core.config import get_llm
from observability.callbacks import build_callback_config

# CONNECTION: The conversation memory logic lives in the root memory module.
# ConversationMemory encapsulates the summarisation policy:
#   should_summarise(messages) — check if history exceeds the threshold
#   summarise_history(messages, old_summary, llm) — compress old messages
#   maybe_summarise(messages, old_summary, llm) — one-step conditional
# This script demonstrates HOW to wire ConversationMemory into a LangGraph
# graph with conditional routing — the pattern, not the implementation.
# See memory/conversation_memory.py for the concept explanation and
# alternative strategies (token-based, importance scoring, selective forget).
from memory.conversation_memory import ConversationMemory


# ============================================================
# STAGE 4.1 — Configuration
# ============================================================
# CONNECTION: ConversationMemory from memory.conversation_memory manages
# the summarisation policy. The thresholds below configure the instance.
#
# In production, SUMMARISE_AFTER would be based on token count (not message
# count) using tiktoken. Message counting is used here for clarity.
# See memory/conversation_memory.py for the alternative strategy discussion.

HISTORY_WINDOW = 6   # keep the last 6 messages (3 turns)
SUMMARISE_AFTER = 4  # trigger summarisation when history exceeds this

# Shared ConversationMemory instance for this pattern script
_conv_memory = ConversationMemory(
    summarise_after=SUMMARISE_AFTER,
    history_window=HISTORY_WINDOW,
)


# ============================================================
# STAGE 4.2 — State Definition
# ============================================================

class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]  # Full conversation history
    summary: str                              # Rolling summary of old messages
    turn_count: int                           # Number of user turns
    needs_summary: bool                       # Flag for summarisation routing


# ============================================================
# STAGE 4.3 — Node Definitions
# ============================================================

def check_history_node(state: ConversationState) -> dict:
    """
    Check if conversation history needs summarisation.

    CONNECTION: Uses ConversationMemory.should_summarise() from the root
    module. This delegates the "is history too long?" decision to the
    root module policy. The threshold is configured in _conv_memory above.

    If True, the router directs to the summarise node before the agent.
    """
    message_count = len(state.get("messages", []))
    current_turn = state.get("turn_count", 0)

    # CONNECTION: should_summarise() checks len(messages) > summarise_after
    needs_summarisation = _conv_memory.should_summarise(state.get("messages", []))
    print(f"    | [Check] Messages: {message_count}, Turn: {current_turn}, Needs summary: {needs_summarisation}")

    return {"needs_summary": needs_summarisation}


def route_after_check(state: ConversationState) -> Literal["summarise", "agent"]:
    """Route to summarise if history is too long, else straight to agent."""
    if state.get("needs_summary", False):
        return "summarise"
    return "agent"


def summarise_node(state: ConversationState) -> dict:
    """
    Compress old messages into a rolling summary using the root module.

    CONNECTION: Uses ConversationMemory.summarise_history() from the root
    module. The method splits messages into old/recent, builds the LLM
    prompt, calls the LLM, and returns:
        (new_message_list, new_summary_text)
    where new_message_list = [SystemMessage(summary)] + recent_messages.

    This node keeps the GRAPH WIRING logic (what to write to state) while
    delegating the SUMMARISATION LOGIC to the root module.
    See memory/conversation_memory.py for the rolling summarisation concept.
    """
    messages = state.get("messages", [])
    old_summary = state.get("summary", "")

    if len(messages) <= HISTORY_WINDOW:
        return {}  # Nothing to summarise (guard against unexpected call)

    llm = get_llm()

    # CONNECTION: summarise_history() handles splitting, prompting, and LLM call.
    # Returns (new_message_list, new_summary_text) ready to write to state.
    new_messages, new_summary = _conv_memory.summarise_history(
        messages=messages,
        old_summary=old_summary,
        llm=llm,
    )

    old_count = len(messages) - len(new_messages) + 1  # +1 for the SystemMessage
    print(f"    | [Summarise] Compressed old messages into {len(new_summary)} char summary")
    print(f"    |   Summary: {new_summary[:100]}...")

    return {
        "messages": new_messages,
        "summary": new_summary,
        "needs_summary": False,
    }


def agent_node(state: ConversationState) -> dict:
    """
    Clinical agent — responds to the latest user message.

    Uses the full message history (or summary + recent) as context.
    """
    llm = get_llm()

    system = SystemMessage(content=(
        "You are a clinical assistant. Answer the patient's questions "
        "about their health clearly and concisely. Reference previous "
        "conversation context when relevant. Keep responses under 100 words."
    ))

    # Build messages for LLM
    conversation = [system] + state["messages"]

    config = build_callback_config(trace_name="conversation_agent")
    response = llm.invoke(conversation, config=config)

    current_turn = state.get("turn_count", 0) + 1
    print(f"    | [Agent] Response: {len(response.content)} chars (turn {current_turn})")

    return {
        "messages": [response],
        "turn_count": current_turn,
    }


# ============================================================
# STAGE 4.4 — Graph Construction
# ============================================================

def build_conversation_graph():
    """
    Build the conversation memory graph with MemorySaver.

    MemorySaver persists conversation state across invoke() calls.
    Each call adds a user message and gets an agent response.
    """
    workflow = StateGraph(ConversationState)

    workflow.add_node("check_history", check_history_node)
    workflow.add_node("summarise", summarise_node)
    workflow.add_node("agent", agent_node)

    workflow.add_edge(START, "check_history")
    workflow.add_conditional_edges(
        "check_history",
        route_after_check,
        {"summarise": "summarise", "agent": "agent"},
    )
    workflow.add_edge("summarise", "agent")
    workflow.add_edge("agent", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# STAGE 4.5 — Main
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  CONVERSATION MEMORY")
    print("  Pattern: multi-turn with rolling summarisation")
    print("=" * 70)

    print(f"""
    Multi-turn conversation with history management:

      Turn 1: User message → Agent response
      Turn 2: User message → Agent response (sees turn 1)
      Turn 3: User message → Agent response (sees turns 1-2)
      Turn 4: History > {SUMMARISE_AFTER} messages → SUMMARISE old turns
              → Agent response (sees summary + recent turns)

    Config:
      SUMMARISE_AFTER = {SUMMARISE_AFTER} messages (trigger threshold)
      HISTORY_WINDOW  = {HISTORY_WINDOW} messages (keep recent)
    """)

    graph = build_conversation_graph()
    config = {"configurable": {"thread_id": "conv-demo-001"}}
    print("    Graph compiled with MemorySaver.\n")

    # ── Simulated multi-turn conversation ─────────────────────────────
    turns = [
        "I'm a 71-year-old woman with CKD Stage 3a and CHF. My potassium came back at 5.4 mEq/L. Should I be worried?",
        "I'm currently taking Lisinopril 20mg and Spironolactone 25mg. Could these be causing the high potassium?",
        "My doctor mentioned reducing the Lisinopril. What dose should I expect, and should I stop the Spironolactone?",
        "I also take Metformin 500mg twice daily for diabetes. Is that safe with my kidney function at eGFR 42?",
        "One more question — I've been having ankle edema and shortness of breath when lying flat. Are these related to my heart failure?",
    ]

    for i, user_msg in enumerate(turns):
        print(f"\n    {'=' * 60}")
        print(f"    TURN {i + 1}")
        print(f"    {'=' * 60}")
        print(f"    User: {user_msg[:80]}{'...' if len(user_msg) > 80 else ''}\n")

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=config,
        )

        # Show the agent's response
        last_ai_message = None
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        if last_ai_message:
            print(f"\n    Agent: {last_ai_message.content[:200]}{'...' if len(last_ai_message.content) > 200 else ''}")
        print(f"\n    Message count: {len(result['messages'])}")
        print(f"    Turn count: {result.get('turn_count', 0)}")
        if result.get("summary"):
            print(f"    Summary: {result['summary'][:100]}...")

    # ── Final state inspection ────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  CONVERSATION MEMORY COMPLETE")
    print("=" * 70)

    snapshot = graph.get_state(config)
    final = snapshot.values

    print(f"""
    Final state:
      Total messages: {len(final.get('messages', []))}
      Turn count: {final.get('turn_count', 0)}
      Summary present: {bool(final.get('summary'))}

    What you saw:
      1. add_messages reducer accumulates messages across turns
      2. MemorySaver persists state between invoke() calls
      3. Rolling summarisation compresses old messages
      4. Agent sees summary + recent messages (bounded context)
      5. Early turn context influences late turn responses

    Token management strategies:
      Message count window  — simple, shown here
      Token count window    — more precise, count with tiktoken
      Importance scoring    — keep high-importance messages longer
      Selective forgetting  — drop low-value exchanges

    Next: shared_memory_multi_agent.py — all memory tiers combined.
    """)


if __name__ == "__main__":
    main()
