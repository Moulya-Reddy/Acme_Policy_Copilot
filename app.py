"""
Streamlit UI for Acme Corp's Policy & IT Copilot.

Shows the agent's live tool-use trace (which tools it called, with what
arguments, and what came back) alongside the final answer - same philosophy
as the Agentic Task Assistant app: transparency into the reasoning, not just
the final output.

Everything in this app is free to run:
- Groq API: free tier (get a key at console.groq.com, no credit card).
- Embeddings: local HuggingFace model, no API cost.
- Vector store: local Chroma, no cloud account.
- Streamlit: free, open-source.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph, MAX_STEPS  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage  # noqa: E402

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Acme Policy Copilot", page_icon="🗂️", layout="wide")
st.title("🗂️ Acme Corp — Policy & IT Copilot")
st.caption(
    "A RAG + tool-calling agent over Acme's HR and IT policy docs, built with "
    "LangChain + LangGraph + a local Chroma index + free Llama 3.3 70B via Groq."
)

with st.sidebar:
    st.header("Setup")
    provider = st.radio(
        "Model provider",
        options=["ollama", "groq"],
        index=0,
        format_func=lambda p: "Ollama (real Llama, local, free forever)" if p == "ollama"
        else "Groq (openai/gpt-oss-120b, hosted, free tier)",
        help=(
            "Groq deprecated its Llama models in Aug 2026 — see the sidebar note below. "
            "Ollama runs an actual Llama model on your Mac with no API key and no deprecation risk."
        ),
    )
    api_key = None
    if provider == "groq":
        api_key = st.text_input(
            "Groq API key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Free at console.groq.com — no credit card required. Not logged or stored.",
        )
        st.caption("⚠️ Groq no longer hosts a general-purpose Llama chat model "
                   "(all deprecated Aug 2026) — this uses openai/gpt-oss-120b instead.")
    else:
        st.caption("Requires Ollama running locally with the model pulled once:\n\n"
                   "`ollama pull llama3.1:8b`")

    st.divider()
    st.header("Available tools")
    st.markdown("**search_policy_docs** — RAG lookup over HR/IT policy docs")
    st.markdown("**check_leave_balance** — mock HR system lookup by employee ID")
    st.markdown("**create_it_ticket** — mock IT ticketing system")
    st.markdown("**calculate** — safe arithmetic (AST-based, no eval())")

    st.divider()
    index_exists = (ROOT / "chroma_db").exists()
    if index_exists:
        st.success("Policy index found.")
    else:
        st.error("No policy index found. Run `python ingest/build_index.py` first.")

    st.divider()
    st.markdown("**Try:**")
    examples = [
        "How many annual leave days do I get, and how many carry over?",
        "What's employee E1003's leave balance?",
        "My VPN keeps disconnecting every hour, please file a ticket, severity 2.",
        "Does Acme offer unlimited paid vacation?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["task_input"] = ex

task = st.text_area("Ask about policy, your leave balance, or file a ticket:",
                     key="task_input", height=100)
run_clicked = st.button("Ask Copilot", type="primary")

if run_clicked:
    if provider == "groq" and not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif not index_exists:
        st.error("Run `python ingest/build_index.py` first to build the policy index.")
    elif not task.strip():
        st.warning("Please enter a question.")
    else:
        app = build_graph(api_key=api_key, provider=provider)
        with st.spinner("Agent working..."):
            result = app.invoke(
                {"messages": [("user", task)], "steps": 0},
                config={"recursion_limit": MAX_STEPS * 2 + 2},
            )

        messages = result["messages"]

        st.subheader("Reasoning trace")
        step_num = 0
        for m in messages:
            if isinstance(m, HumanMessage):
                continue
            if isinstance(m, AIMessage) and m.tool_calls:
                step_num += 1
                with st.expander(f"Step {step_num}: model requests tool call(s)", expanded=False):
                    for tc in m.tool_calls:
                        st.markdown(f"**Tool:** `{tc['name']}`")
                        st.json(tc["args"])
            elif isinstance(m, ToolMessage):
                with st.expander(f"Observation from `{m.name}`", expanded=False):
                    st.code(m.content, language=None)
            elif isinstance(m, AIMessage) and m.content:
                pass  # the final answer is rendered below, not in the trace

        st.divider()
        st.subheader("Answer")
        final_answer = messages[-1].content if messages else "(no answer)"
        st.success(final_answer)

        tools_called = [
            tc["name"] for m in messages if isinstance(m, AIMessage) and m.tool_calls for tc in m.tool_calls
        ]
        cols = st.columns(3)
        cols[0].metric("Tool calls made", len(tools_called))
        cols[1].metric("Reasoning steps", result.get("steps", 0))
        cols[2].metric("Tools used", ", ".join(sorted(set(tools_called))) or "none")

        if result.get("steps", 0) >= MAX_STEPS:
            st.warning("Agent hit its step limit before reaching a confident final answer.")
