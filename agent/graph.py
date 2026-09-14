"""
LangGraph orchestration for the Policy Copilot agent.

This is the LangGraph equivalent of the Agentic Task Assistant's hand-rolled
`Agent.run()` loop (see agent/core.py in that project) - same underlying
idea (call the model, see if it wants a tool, run the tool, feed the result
back, repeat until a final answer), expressed as an explicit graph instead
of a Python `for` loop, and using NATIVE tool calling (the model returns a
structured, schema-validated tool_call) instead of regex-parsed plain text.

    ┌─────────────┐
    │    agent     │  <- calls the LLM (with tools bound); may emit tool_calls
    └──────┬──────┘
           │
   conditional edge: does the last AIMessage have tool_calls?
           │
     ┌─────┴─────┐
   YES           NO
     │             │
     ▼             ▼
 ┌───────┐        END
 │ tools  │  <- ToolNode executes the requested tool(s), appends ToolMessage(s)
 └───┬───┘
     │
     └──────────► back to agent (loop)

A step counter in state enforces a hard maximum, exactly like MAX_STEPS in
the from-scratch ReAct agent - this guarantees termination and bounds
worst-case cost/latency no matter what the model does.
"""

from __future__ import annotations
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.tools import ALL_TOOLS
from agent.llm import get_llm

MAX_STEPS = 6

SYSTEM_PROMPT = """You are Acme Corp's internal Policy & IT Copilot.

You help employees with questions about HR policy, IT policy, their own \
leave balance, and filing IT tickets.

Rules:
- For ANY question about company policy (leave rules, expense limits, \
password rules, ticket severity levels, etc.), use the search_policy_docs \
tool rather than answering from memory - your training data does not \
contain Acme Corp's actual, current policies.
- For a specific employee's leave balance, use check_leave_balance.
- Only use create_it_ticket if the user is actually asking you to file a \
ticket, not just asking about IT policy.
- Use calculate for any arithmetic instead of computing it yourself.
- If a tool returns "no relevant information", say so plainly rather than \
guessing an answer.
- Keep answers concise and cite which policy section your answer is based on \
when you used search_policy_docs.
"""


class AgentState(TypedDict):
    # `add_messages` is a LangGraph reducer: each node returns messages to
    # APPEND, rather than having to manually manage the full message list -
    # the graph merges them into the running conversation state for you.
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


def build_graph(api_key: str | None = None, provider: str | None = None):
    llm = get_llm(api_key=api_key, provider=provider)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        # Only prepend the system prompt once, on the first call.
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if state.get("steps", 0) >= MAX_STEPS:
            return "end"  # hard safety cap, same role as MAX_STEPS in the from-scratch agent
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent(question: str, api_key: str | None = None, provider: str | None = None) -> dict:
    """Runs the compiled graph on a single question and returns a dict with
    the final answer text and the full message trace (for the Streamlit
    trace view and for evaluation)."""
    app = build_graph(api_key=api_key, provider=provider)
    result = app.invoke(
        {"messages": [("user", question)], "steps": 0},
        config={"recursion_limit": MAX_STEPS * 2 + 2},
    )
    messages = result["messages"]
    final_answer = messages[-1].content if messages else ""
    return {"answer": final_answer, "messages": messages, "steps": result.get("steps", 0)}
