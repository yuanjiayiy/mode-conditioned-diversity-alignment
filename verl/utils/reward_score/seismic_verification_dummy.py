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
"""Dummy reward for seismic verification.

Only returns reward 1 when the parsed label is exactly "FALSE POSITIVE".
"""

from __future__ import annotations

from typing import Any

from .seismic_verification import normalize_class_label, parse_model_reasoning, parse_predicted_label


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return 1 only for "FALSE POSITIVE" predictions; otherwise return 0."""
    _ = ground_truth
    _ = extra_info

    pred_label = parse_predicted_label(solution_str)
    reasoning_text = parse_model_reasoning(solution_str)

    if pred_label is None:
        return {
            "score": 0.0,
            "acc": False,
            "error": "invalid_prediction",
            "pred_label": None,
            "gt_label": None,
            "reason": reasoning_text,
        }

    if normalize_class_label(pred_label) == "false positive":
        return {
            "score": 1.0,
            "acc": True,
            "error": None,
            "pred_label": pred_label,
            "gt_label": None,
            "reason": reasoning_text,
        }

    return {
        "score": 0.0,
        "acc": False,
        "error": "not_false_positive",
        "pred_label": pred_label,
        "gt_label": None,
        "reason": reasoning_text,
    }
