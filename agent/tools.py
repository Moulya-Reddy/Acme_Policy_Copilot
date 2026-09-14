"""
Tools available to the agent.

All free, all local, no external paid API for any tool:
- search_policy_docs   -> RAG lookup against the local Chroma index (Section: RAG)
- check_leave_balance   -> a mocked HR system lookup (a real enterprise version
                            would call an internal HRIS API - Workday, SAP
                            SuccessFactors, etc. - this stands in for that)
- create_it_ticket       -> a mocked IT ticketing system write (a real version
                             would call ServiceNow/Jira Service Desk's API)
- calculate               -> a safe, AST-based arithmetic tool (same security
                             reasoning as the Agentic Task Assistant project:
                             never eval() untrusted/model-generated strings)

Each tool is a plain Python function decorated with @tool from LangChain,
which auto-generates the JSON schema the LLM needs for native tool calling
from the function's type hints and docstring - this replaces the manual
system-prompt-text tool descriptions used in the from-scratch ReAct project.
"""

from __future__ import annotations
import ast
import operator
import json
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from rag.retriever import retrieve, format_context

ROOT = Path(__file__).resolve().parent.parent
TICKETS_FILE = ROOT / "data" / "tickets.json"

# ---------------------------------------------------------------------------
# Mock enterprise "systems of record" - in-memory / local-file stand-ins for
# what would be real internal API calls in production. Kept obviously fake
# and clearly labelled so this is safe to demo without connecting anything.
# ---------------------------------------------------------------------------

_MOCK_LEAVE_DB = {
    "E1001": {"name": "Asha Rao", "annual_leave_balance": 9.5, "sick_leave_balance": 12},
    "E1002": {"name": "Vikram Shah", "annual_leave_balance": 3.0, "sick_leave_balance": 7},
    "E1003": {"name": "Priya Menon", "annual_leave_balance": 14.0, "sick_leave_balance": 12},
}


@tool
def search_policy_docs(query: str) -> str:
    """Search Acme Corp's HR and IT policy documents for information relevant
    to the query. Use this for any question about company policy - leave
    rules, expense limits, IT security rules, ticket severity levels, etc.
    Input: a natural-language question or topic, e.g. 'how many sick days do
    employees get' or 'password requirements'."""
    docs = retrieve(query, k=4)
    if not docs:
        return "No relevant policy sections found for this query."
    return format_context(docs)


@tool
def check_leave_balance(employee_id: str) -> str:
    """Look up an employee's current leave balances by their employee ID
    (format: E followed by 4 digits, e.g. 'E1001'). This is a MOCK internal
    HR system for demo purposes - it does not access any real employee data.
    Input: an employee ID string."""
    record = _MOCK_LEAVE_DB.get(employee_id.strip().upper())
    if not record:
        return (
            f"No record found for employee ID '{employee_id}'. "
            f"Valid demo IDs: {', '.join(_MOCK_LEAVE_DB.keys())}"
        )
    return (
        f"{record['name']} ({employee_id.upper()}): "
        f"{record['annual_leave_balance']} annual leave days remaining, "
        f"{record['sick_leave_balance']} sick leave days remaining."
    )


@tool
def create_it_ticket(summary: str, severity: Literal["1", "2", "3", "4"] = "3") -> str:
    """Create an IT support ticket. Use this when the user wants to actually
    report an issue (not just ask about IT policy - use search_policy_docs
    for that). Severity must be '1' (critical) through '4' (low) per the IT
    policy's severity definitions. This is a MOCK ticketing system for demo
    purposes - it writes to a local JSON file, not a real ticketing system.
    Input: a short summary of the issue, and an optional severity level."""
    # Restricting severity to a fixed Literal set (rather than accepting any
    # free-text string) is a small but real security/robustness choice: it
    # stops the tool from being called with an arbitrary, unvalidated value.
    tickets = []
    if TICKETS_FILE.exists():
        tickets = json.loads(TICKETS_FILE.read_text())

    ticket_id = f"TCK-{1000 + len(tickets) + 1}"
    tickets.append({"ticket_id": ticket_id, "summary": summary, "severity": severity})
    TICKETS_FILE.write_text(json.dumps(tickets, indent=2))

    return f"Ticket {ticket_id} created (severity {severity}): '{summary}'"


# ---------------------------------------------------------------------------
# Safe calculator - AST-based, same reasoning as the Agentic Task Assistant
# project's calculator: never eval() a string that could ultimately trace
# back to LLM or user-influenced input.
# ---------------------------------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression.")


@tool
def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression, e.g. for computing leave
    days remaining after a request, or days until a deadline. Input: a math
    expression string, e.g. '18 - 9.5' or '(30 - 5) * 2'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


ALL_TOOLS = [search_policy_docs, check_leave_balance, create_it_ticket, calculate]
