# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rule-based reward for seismic event verification (RL²F / PPO).

Supported completions (first match wins):

1. ``<answer>...</answer>`` with one of ``SEISMIC_OUTPUT_LABELS`` inside.
2. Structured lines::

     VERDICT: TRUE SEISMIC EVENT | FALSE POSITIVE
     SEISMIC_EVENT_TYPE: EARTHQUAKE | ...   (optional; underscores may appear as ``\\_`` in text)
     Reasoning: ...

   Verdict + event type are mapped to the same canonical ``pred_label`` as expert CSV fields.
3. Legacy: a single line matching an allowed label.

Ground truth comes from ``expert_majority_vote_label`` and ``expert_event_type``.

Scoring:
  * ``+1`` — prediction matches expert majority verdict and event type.
  * ``0`` — prediction is a valid label but wrong.
  * ``-1`` — prediction is not one of the allowed labels (invalid / unparsable).

Return dict includes ``pred_label``, ``reason`` (text after ``Reasoning:`` when present), and
``error`` for machine-readable failure tags (``invalid_prediction``, ``wrong_label``,
``undefined_ground_truth``) when applicable.
"""

from __future__ import annotations

import re
from typing import Any

# Canonical labels (lowercase). Prompt instructs the model to output exactly one of these.
SEISMIC_OUTPUT_LABELS: tuple[str, ...] = (
    "false positive",
    "true seismic, earthquake",
    "true seismic, non-earthquake",
)
ALLOWED_LABELS: frozenset[str] = frozenset(SEISMIC_OUTPUT_LABELS)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_class_label(s: str) -> str:
    """Lowercase and collapse whitespace for comparison."""
    return _norm_ws(s).lower()


def parse_ground_truth_verdict(majority_label: str) -> str | None:
    """Return 'true_seismic' | 'false_positive' | None."""
    t = normalize_class_label(majority_label)
    if not t:
        return None
    if "false" in t and "positive" in t:
        return "false_positive"
    if "true" in t and "seismic" in t:
        return "true_seismic"
    return None


def parse_ground_truth_event_type(expert_event_type: str | None, verdict: str | None) -> str | None:
    """Return 'earthquake' | 'non_earthquake' | 'na' | None (missing subtype for true seismic)."""
    if verdict == "false_positive" or verdict is None:
        return "na"
    if expert_event_type is None:
        return None
    s = _norm_ws(str(expert_event_type)).lower()
    if not s or s in ("nan", "none"):
        return None
    if "non" in s and "earthquake" in s:
        return "non_earthquake"
    if "earthquake" in s:
        return "earthquake"
    return None


def canonical_expert_label(majority_label: str, expert_event_type: str | None) -> str | None:
    """Map CSV fields to exactly one of ``SEISMIC_OUTPUT_LABELS``, or None if undefined."""
    v = parse_ground_truth_verdict(majority_label)
    if v == "false_positive":
        return "FALSE POSITIVE"
    if v != "true_seismic":
        return None
    t = parse_ground_truth_event_type(expert_event_type, v)
    if t == "earthquake":
        return "TRUE SEISMIC EVENT, EARTHQUAKE"
    if t == "non_earthquake":
        return "TRUE SEISMIC EVENT, NON-EARTHQUAKE"
    return None


def parse_model_reasoning(solution_str: str) -> str | None:
    """Return text after the first ``Reasoning:`` (rest of response, trimmed)."""
    if not solution_str:
        return None
    m = re.search(r"(?is)(?:\*\*)?Reasoning(?:\*\*)?\s*:\s*(.*)\Z", solution_str.strip())
    if not m:
        return None
    t = m.group(1).strip()
    return t if t else None


_VERDICT_LINE = re.compile(r"(?im)^\s*(?:\*\*)?VERDICT(?:\*\*)?\s*:\s*([^\n\r]+?)\s*$")
# Matches SEISMIC_EVENT_TYPE or SEISMIC\_EVENT\_TYPE (escaped underscores from some models)
_SEISMIC_TYPE_LINE = re.compile(
    r"(?im)^\s*(?:\*\*)?SEISMIC(?:\\_|_)EVENT(?:\\_|_)TYPE(?:\*\*)?\s*:\s*([^\n\r]+?)\s*$"
)


def _label_from_verdict_format(solution_str: str) -> str | None:
    """
    Map ``VERDICT:`` + optional ``SEISMIC_EVENT_TYPE:`` lines to a canonical label, or None.
    """
    if not solution_str:
        return None
    vm = _VERDICT_LINE.search(solution_str)
    if not vm:
        return None
    verdict_raw = vm.group(1).strip()
    v = normalize_class_label(verdict_raw)

    tm = _SEISMIC_TYPE_LINE.search(solution_str)
    event_raw = tm.group(1).strip() if tm else ""
    et = normalize_class_label(event_raw) if event_raw else ""

    if "false" in v and "positive" in v:
        return "FALSE POSITIVE"
    if "true" in v and "seismic" in v:
        if "non" in et and "earthquake" in et:
            return "TRUE SEISMIC EVENT, NON-EARTHQUAKE"
        if "earthquake" in et:
            return "TRUE SEISMIC EVENT, EARTHQUAKE"
        return None
    return None


def _extract_answer_span(solution_str: str) -> str | None:
    """Return raw inner text of the first <answer>...</answer> block, or None."""
    if not solution_str:
        return None
    m = re.search(
        r"<\s*answer\s*>([\s\S]*?)<\s*/\s*answer\s*>",
        solution_str,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip()


def parse_predicted_label(solution_str: str) -> str | None:
    """
    If the response contains an ``<answer>`` open tag, parse **only** the matching
    ``<answer>...</answer>`` block (normalized label must be in ``ALLOWED_LABELS``).
    Else if ``VERDICT:`` is present, map verdict (+ ``SEISMIC_EVENT_TYPE``) to a canonical label.
    Otherwise fall back to legacy parsing (plain line / whole text).
    """
    if not solution_str:
        return None
    if re.search(r"<\s*answer\s*>", solution_str, flags=re.IGNORECASE):
        inner = _extract_answer_span(solution_str)
        if inner is None:
            return None
        cand = normalize_class_label(inner)
        return cand if cand in ALLOWED_LABELS else None

    verdict_label = _label_from_verdict_format(solution_str)
    if verdict_label is not None:
        return verdict_label

    whole = normalize_class_label(solution_str)
    if whole in ALLOWED_LABELS:
        return whole
    for line in solution_str.splitlines():
        ln = normalize_class_label(line)
        if not ln:
            continue
        if ln in ALLOWED_LABELS:
            return ln
        if ":" in line:
            tail = normalize_class_label(line.rsplit(":", 1)[-1])
            if tail in ALLOWED_LABELS:
                return tail
    return None

def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Args:
        solution_str: Model completion (``<answer>`` should contain one of ``SEISMIC_OUTPUT_LABELS``).
        ground_truth: ``expert_majority_vote_label``.
        extra_info: Should include ``expert_event_type`` from CSV when applicable.

    Returns:
        Dict with ``score`` in ``{-1, 0, 1}``, ``acc`` True iff score == 1.
    """
    extra_info = extra_info or {}
    expert_type = extra_info.get("expert_event_type")
    if expert_type is not None:
        try:
            import math

            if isinstance(expert_type, float) and math.isnan(expert_type):
                expert_type = None
        except (TypeError, ValueError):
            pass
        if expert_type is not None:
            s = str(expert_type).strip()
            expert_type = None if not s or s.lower() in ("nan", "none") else s

    gt_label = canonical_expert_label(ground_truth, expert_type)
    pred_label = parse_predicted_label(solution_str)
    reasoning_text = parse_model_reasoning(solution_str)
    
    print(f"GT label: {gt_label}, Pred label: {pred_label}, Reasoning: {reasoning_text}")

    if gt_label is None:
        return {
            "score": -1.0,
            "acc": False,
            "error": "undefined_ground_truth",
            "pred_label": pred_label,
            "gt_label": None,
            "reason": reasoning_text,
        }

    if pred_label is None:
        return {
            "score": -1.0,
            "acc": False,
            "error": "invalid_prediction",
            "pred_label": None,
            "gt_label": gt_label,
            "reason": reasoning_text,
        }

    if pred_label.lower() == gt_label.lower():
        return {
            "score": 1.0,
            "acc": True,
            "error": None,
            "pred_label": pred_label,
            "gt_label": gt_label,
            "reason": reasoning_text,
        }

    return {
        "score": 0.0,
        "acc": False,
        "error": "wrong_label",
        "pred_label": pred_label,
        "gt_label": gt_label,
        "reason": reasoning_text,
    }


