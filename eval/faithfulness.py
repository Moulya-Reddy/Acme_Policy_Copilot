"""
Lightweight, deterministic faithfulness check: does every number the agent's
final answer states also appear somewhere in the retrieved policy text it
was given? Same approach and same rationale as RivalIQ's eval/faithfulness.py
- for content where the claims that matter are exact numbers (leave days,
password length, severity levels), matching numbers against ground truth is
a cheaper and more reproducible signal than an LLM-judge.

This is intentionally narrow: it only catches NUMERIC hallucination, not
qualitative misstatements. See the README's "Evaluation" section for how
you'd extend this with an LLM-judge for broader answer-relevance scoring.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

_NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*")


def _extract_numbers(text: str) -> List[float]:
    return [float(n) for n in _NUMBER_RE.findall(text) if n not in ("", "-", "+")]


@dataclass
class FaithfulnessResult:
    total_numeric_claims: int
    grounded_claims: int
    ungrounded_values: List[float]

    @property
    def faithfulness_score(self) -> float:
        if self.total_numeric_claims == 0:
            return 1.0
        return round(self.grounded_claims / self.total_numeric_claims, 3)


def check_faithfulness(answer: str, retrieved_context: str, tolerance: float = 0.05) -> FaithfulnessResult:
    ground_truth_numbers = set(_extract_numbers(retrieved_context))
    claimed_numbers = _extract_numbers(answer)

    grounded, ungrounded = 0, []
    for num in claimed_numbers:
        if any(abs(num - gt) <= tolerance for gt in ground_truth_numbers):
            grounded += 1
        else:
            ungrounded.append(num)

    return FaithfulnessResult(
        total_numeric_claims=len(claimed_numbers),
        grounded_claims=grounded,
        ungrounded_values=ungrounded,
    )
