"""
Evaluation harness for the Policy Copilot agent.

Runs a fixed set of benchmark tasks (eval/test_tasks.json) through the full
LangGraph agent and reports, per task:
  - which tool(s) were actually used, vs. which were expected
  - the faithfulness score of the final answer (for RAG-backed answers)
  - latency, input/output tokens
  - the answer itself, for manual correctness review

Usage:
    export GROQ_API_KEY=gsk_...
    python eval/run_eval.py

Everything here runs on Groq's free tier - $0 cost as long as you stay
within the free-tier rate limits.
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from agent.graph import build_graph, MAX_STEPS  # noqa: E402
from eval.faithfulness import check_faithfulness  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_tasks():
    with open(HERE / "test_tasks.json") as f:
        return json.load(f)


def extract_tool_usage(messages):
    """Returns (tools_used: list[str], retrieved_context: str) from the
    message trace. `retrieved_context` concatenates every search_policy_docs
    tool result, since that's the ground truth the faithfulness check needs."""
    tools_used = []
    retrieved_chunks = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tools_used.extend(tc["name"] for tc in m.tool_calls)
        if isinstance(m, ToolMessage) and m.name == "search_policy_docs":
            retrieved_chunks.append(m.content)
    return tools_used, "\n".join(retrieved_chunks)


def main():
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    if provider == "groq" and not os.environ.get("GROQ_API_KEY"):
        print("Error: set GROQ_API_KEY (free at console.groq.com), or set "
              "LLM_PROVIDER=ollama to use a local model instead.")
        sys.exit(1)

    tasks = load_tasks()
    app = build_graph(provider=provider)
    results = []

    print(f"Running {len(tasks)} eval tasks...\n")
    for t in tasks:
        print(f"[{t['id']}] {t['task']}")
        start = time.time()
        result = app.invoke(
            {"messages": [("user", t["task"])], "steps": 0},
            config={"recursion_limit": MAX_STEPS * 2 + 2},
        )
        latency = time.time() - start
        messages = result["messages"]
        answer = messages[-1].content if messages else ""

        tools_used, retrieved_context = extract_tool_usage(messages)

        faith = None
        if "search_policy_docs" in tools_used:
            faith = check_faithfulness(answer, retrieved_context)

        # Token usage: LangChain's ChatOpenAI attaches usage_metadata to each
        # AIMessage when the provider returns it (Groq's OpenAI-compatible
        # endpoint does). Summed across every model call in this task's run.
        input_tokens = sum(
            getattr(m, "usage_metadata", {}).get("input_tokens", 0)
            for m in messages if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None)
        )
        output_tokens = sum(
            getattr(m, "usage_metadata", {}).get("output_tokens", 0)
            for m in messages if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None)
        )

        record = {
            "id": t["id"],
            "task": t["task"],
            "expected_tool": t["expects_tool"],
            "tools_used": tools_used,
            "answer": answer,
            "faithfulness_score": faith.faithfulness_score if faith else None,
            "latency_seconds": round(latency, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "steps": result.get("steps", 0),
        }
        results.append(record)

        print(f"  -> tools used: {tools_used or 'none'}")
        print(f"  -> faithfulness: {record['faithfulness_score']}")
        print(f"  -> {latency:.1f}s, {input_tokens} in / {output_tokens} out tokens\n")

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    faith_scores = [r["faithfulness_score"] for r in results if r["faithfulness_score"] is not None]
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else None

    print("=" * 60)
    print(f"Average latency: {avg_latency:.1f}s per task")
    if avg_faith is not None:
        print(f"Average faithfulness (RAG tasks only): {avg_faith:.1%}")
    print(f"API cost: $0.00 ({'local Ollama, no network cost' if provider == 'ollama' else 'Groq free tier'})")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
