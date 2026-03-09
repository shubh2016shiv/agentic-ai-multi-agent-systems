#!/usr/bin/env python3
"""
============================================================
ToolNode Patterns: Graph Node vs Manual Invocation
============================================================
Prerequisite: tool_binding.py

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------
LangGraph offers two ways to execute tools. This script
builds both patterns side by side so you can compare them.

Pattern 1 (Graph Node):
    ToolNode is registered as a named node in the graph.
    The agent node produces tool calls. A conditional edge
    routes to the ToolNode. ToolNode results go back to the
    agent via another edge. Repeat until no tool calls.

Pattern 2 (Manual Invocation):
    ToolNode.invoke() is called inside the agent's node
    function. The loop is explicit in Python code.
    No extra nodes or edges.

Both produce the same result. The choice depends on whether
you want tool execution visible in the graph topology
(Pattern 1) or encapsulated within a node (Pattern 2).

------------------------------------------------------------
DIAGRAMS
------------------------------------------------------------

    PATTERN 1 — ToolNode as Graph Node:

        [START]
           |
           v
        [agent] ─── has_tool_calls? ──> [tools]
           ^                               |
           |                               |
           +---------- results ───────────+
           |
           +── no tool calls ──> [END]


    PATTERN 2 — Manual ToolNode.invoke():

        [START]
           |
           v
        [agent]    <-- contains internal loop:
           |            response = llm.invoke()
           |            while response.tool_calls:
           |                results = ToolNode.invoke()
           |                response = llm.invoke()
           |
           v
         [END]

------------------------------------------------------------
WHAT YOU LEARN
------------------------------------------------------------
    1. Pattern 1: ToolNode as a separate graph node
    2. Pattern 2: ToolNode.invoke() inside a node function
    3. When to choose each pattern
    4. Stream output comparison

------------------------------------------------------------
WHEN TO USE
------------------------------------------------------------
    Pattern 1 (ToolNode as graph node):
        Use when tool execution should be visible in the graph
        topology — useful for tracing, debugging, and HITL
        confirmation of tool calls.

    Pattern 2 (ToolNode.invoke() inside node):
        Use when you want the tool execution loop encapsulated
        within the agent node — simpler graph topology, but tool
        calls are not separately addressable in the graph.

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
    cd D:/Agentic AI/LangGraph_Multi_Agent_System
    python -m scripts.tools.toolnode_patterns
============================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
import os
import json
from typing import TypedDict, Annotated, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── LangGraph ───────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── Project imports ─────────────────────────────────────────────────────────
# CONNECTION: core/ root module — get_llm() centralises LLM config.
# PatientCase is the canonical domain model used in test scenarios.
from core.config import get_llm
from core.models import PatientCase
# CONNECTION: tools/ root module — analyze_symptoms and assess_patient_risk are
# component-layer tools. This script demos two patterns for INVOKING them
# (ToolNode as node vs ToolNode.invoke()). See tools/ for implementations.
from tools import analyze_symptoms, assess_patient_risk
# CONNECTION: observability/ root module — build_callback_config() attaches
# Langfuse trace_name and tags to every LLM call automatically.
from observability.callbacks import build_callback_config


# ── Shared state ────────────────────────────────────────────────────────────
class ToolDemoState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_response: str


# ── Shared test case ────────────────────────────────────────────────────────
PATIENT = PatientCase(
    patient_id="PT-TN-001",
    age=58, sex="M",
    chief_complaint="Persistent cough and dyspnea for 3 weeks",
    symptoms=["cough", "dyspnea", "wheezing", "fatigue"],
    medical_history=["COPD Stage II"],
    current_medications=["Tiotropium 18mcg inhaler daily"],
    allergies=[],
    lab_results={"FEV1": "58% predicted", "SpO2": "93%"},
    vitals={"BP": "138/85", "HR": "92"},
)

TOOLS = [analyze_symptoms, assess_patient_risk]

# Shared prompt
SYSTEM_MSG = SystemMessage(content="Evaluate the patient using your tools.")
USER_MSG = HumanMessage(content=f"""Patient: {PATIENT.age}y {PATIENT.sex}
Complaint: {PATIENT.chief_complaint}
Symptoms: {', '.join(PATIENT.symptoms)}
Labs: {json.dumps(PATIENT.lab_results)}

Use your tools, then provide your assessment.""")


# ============================================================
# STAGE 7.1 — Pattern 1: ToolNode as Graph Node
# ============================================================

def build_pattern_1():
    """
    Build a graph where ToolNode is a separate named node.

    The agent node produces tool calls. A conditional edge
    checks for tool calls and routes to the "tools" node.
    Results flow back to the agent via an edge.
    """
    llm = get_llm()
    agent_llm = llm.bind_tools(TOOLS)

    def agent_node(state: ToolDemoState) -> dict:
        """Agent node — calls LLM, may produce tool calls."""
        config = build_callback_config(trace_name="toolnode_p1_agent")
        response = agent_llm.invoke(state["messages"], config=config)

        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | [P1 Agent] Tool calls: {[tc['name'] for tc in response.tool_calls]}")
        else:
            print(f"    | [P1 Agent] Final response: {len(response.content)} chars")

        return {"messages": [response], "agent_response": response.content or ""}

    def should_use_tools(state: ToolDemoState) -> Literal["tools", "end"]:
        """Route to ToolNode if the last message has tool calls."""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "end"

    workflow = StateGraph(ToolDemoState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_use_tools,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")  # Results go back to agent

    return workflow.compile()


# ============================================================
# STAGE 7.2 — Pattern 2: Manual ToolNode.invoke()
# ============================================================

def build_pattern_2():
    """
    Build a graph where ToolNode.invoke() is called inside
    the agent node function. No separate "tools" node.
    """
    llm = get_llm()
    agent_llm = llm.bind_tools(TOOLS)

    def agent_node(state: ToolDemoState) -> dict:
        """
        Agent with internal tool loop.

        The ToolNode is created and invoked inside this function.
        Results are fed back to the LLM within the same node.
        """
        config = build_callback_config(trace_name="toolnode_p2_agent")
        messages = list(state["messages"])
        response = agent_llm.invoke(messages, config=config)

        # Internal tool loop
        while hasattr(response, "tool_calls") and response.tool_calls:
            print(f"    | [P2 Agent] Tool calls: {[tc['name'] for tc in response.tool_calls]}")

            tool_node = ToolNode(TOOLS)
            tool_results = tool_node.invoke({"messages": [response]})
            messages.extend([response] + tool_results["messages"])
            response = agent_llm.invoke(messages, config=config)

        print(f"    | [P2 Agent] Final response: {len(response.content)} chars")

        return {"messages": [response], "agent_response": response.content}

    workflow = StateGraph(ToolDemoState)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


# ============================================================
# STAGE 7.3 — Run Both Patterns
# ============================================================

def run_pattern(label: str, graph) -> str:
    """Run a pattern and return the response."""
    print(f"\n    Running {label}...")

    initial_state: ToolDemoState = {
        "messages": [SYSTEM_MSG, USER_MSG],
        "agent_response": "",
    }

    result = graph.invoke(initial_state)
    response = result.get("agent_response", "")

    # Count tool messages
    tool_msgs = [m for m in result["messages"] if hasattr(m, "name") and m.name]
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]

    print(f"    | Total messages : {len(result['messages'])}")
    print(f"    | Tool results   : {len(tool_msgs)}")
    print(f"    | AI messages    : {len(ai_msgs)}")
    print(f"    | Response       : {response[:100]}...")

    return response


# ============================================================
# STAGE 7.4 — Summary
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  TOOLNODE PATTERNS: GRAPH NODE vs MANUAL INVOCATION")
    print("=" * 70)

    print("""
    Two patterns for tool execution in LangGraph:

    PATTERN 1 (Graph Node):              PATTERN 2 (Manual):
    ┌─────────────────────┐              ┌─────────────────────┐
    │ [agent] ─> [tools]  │              │ [agent]             │
    │    ^          |     │              │   (internal loop)   │
    │    +──────────+     │              │                     │
    │    |                │              │                     │
    │    v                │              │                     │
    │  [END]              │              │ [END]               │
    └─────────────────────┘              └─────────────────────┘
    Nodes: 2 (agent, tools)              Nodes: 1 (agent)
    Tool steps visible in stream         Tool steps internal
    """)

    # Build both
    graph_p1 = build_pattern_1()
    graph_p2 = build_pattern_2()

    print("=" * 70)
    print("  PATTERN 1: ToolNode as Graph Node")
    print("=" * 70)
    response_p1 = run_pattern("Pattern 1", graph_p1)

    print("\n\n" + "=" * 70)
    print("  PATTERN 2: Manual ToolNode.invoke()")
    print("=" * 70)
    response_p2 = run_pattern("Pattern 2", graph_p2)

    # Compare
    print("\n\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"""
    Both patterns produce clinical assessments. The difference
    is in graph visibility and control:

    Pattern 1 (Graph Node):
      + Tool execution visible as a separate graph step
      + Better for streaming — caller sees "agent", "tools", "agent"
      + Easier to add pre/post-processing around tool execution
      - More graph nodes and edges
      - Tool loop is implicit in the graph topology

    Pattern 2 (Manual Invocation):
      + Simpler graph — one node does everything
      + Full control over the tool loop (retries, filtering)
      + Encapsulated — caller sees a single "agent" step
      - Tool execution invisible to graph streaming
      - Harder to intercept individual tool calls

    When to use which:
      Pattern 1: when you need streaming, debugging, or
                 middleware between agent and tools.
      Pattern 2: when the agent's tool loop is self-contained
                 and you want simpler graph topology.

    Next: structured_output.py — getting validated JSON from agents.
    """)


if __name__ == "__main__":
    main()
