"""
Shared utility functions for custom reward functions.
Used by: infinite_chat_reward_function, custom_reward_function, openscholar_reward_function.
"""

import re
import os
import logging
import asyncio
import threading
from difflib import SequenceMatcher
from typing import Any, List, Callable, Optional
import time
import numpy as np
import torch
from verl.utils.nlp_utils import _text_after_thinking
from verl.utils.partition import equivalence_check_classifier, equivalence_check_gpt4

logger = logging.getLogger(__name__)

# Constants shared across reward modules
DEFAULT_DEVICE_ENV = "CUSTOM_REWARD_DEVICE"
MATH_SOURCES = {"math", "gsm8k"}
TEXT_SOURCES = {"text", "chat", "instruction"}
SCHOLAR_SOURCES = {"rl-research/dr-tulu-sft-data"}
LEAN_SOURCES = {"lean", "proofnet", "proofnet#", "minif2f", "PAug/ProofNetSharp", "putnambench"}
DEBUG_PRINT_LIMIT = int(os.environ.get("CUSTOM_REWARD_DEBUG_PRINT_N", "0"))


def resolve_device(explicit: str | None = None) -> str:
    """Pick the device for reward models, honoring env overrides."""
    if explicit:
        return explicit
    env_device = os.environ.get(DEFAULT_DEVICE_ENV)
    if env_device:
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def to_scalar_text(value: Any) -> str:
    """Convert tensors/arrays/lists to a single string for prompt/response/ground_truth."""
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.flatten()[0]
    if torch.is_tensor(value):
        value = value.item() if value.numel() == 1 else value.flatten()[0].item()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value)


def is_empty(text: str) -> bool:
    return text is None or len(text.strip()) == 0


def normalize_seq(obj: Any, expected_len: int | None = None) -> list[str]:
    """
    Normalize a possibly numpy array / tensor / scalar into a list of strings.
    If expected_len is provided and obj is scalar, repeat to that length.
    """
    if isinstance(obj, np.ndarray):
        flat = obj.reshape(-1).tolist()
        return [str(x) for x in flat]
    if torch.is_tensor(obj):
        flat = obj.detach().cpu().reshape(-1).tolist()
        return [str(x) for x in flat]
    if isinstance(obj, (list, tuple)):
        return [str(x) for x in obj]
    if expected_len is not None:
        return [str(obj) for _ in range(expected_len)]
    return [str(obj)]

def format_reward_info(
    total_reward: float,
    base_reward: float,
    math_component: float,
    text_component: float,
    math_raw: float,
    text_raw: float,
    quality_gate: float,
    distinctiveness_raw: float,
    distinctiveness_component: float,
    length_penalty_component: float = 0.0,
    language_penalty_component: float = 0.0,
    response_n_tokens: int = 0,
    thinking_n_tokens: int = 0,
    answer_n_tokens: int = 0,
    format_reward: float = 1.0,
    **kwargs,
 ) -> dict:
    """Standardize reward breakdown dict so Naive/Batch managers can log metrics."""
    out = {
        "score": float(total_reward),
        "base_reward": float(base_reward),
        "accuracy_reward": float(math_component),
        "accuracy_raw": float(math_raw),
        "text_reward": float(text_component),
        "text_reward_raw": float(text_raw),
        "quality_gate": float(quality_gate),
        "distinctiveness_raw": float(distinctiveness_raw),
        "distinctiveness_reward": float(distinctiveness_component),
        "distinctiveness_loss": float(-distinctiveness_component),
        "length_penalty": float(length_penalty_component),
        "language_penalty": float(language_penalty_component),
        "response_n_tokens": int(response_n_tokens),
        "thinking_n_tokens": int(thinking_n_tokens),
        "answer_n_tokens": int(answer_n_tokens),
        "format_reward": float(format_reward),
    }
    # Merge any additional fields (extras) into the returned dict. Extras may
    # override existing keys if present — caller responsibility.
    if kwargs:
        for k, v in kwargs.items():
            out[k] = v
    return out


def zero_reward_info(
    length_penalty_component: float = 0.0,
    language_penalty_component: float = 0.0,
    response_n_tokens: int = 0,
) -> dict:
    return format_reward_info(
        total_reward=0.0,
        base_reward=0.0,
        math_component=0.0,
        text_component=0.0,
        math_raw=0.0,
        text_raw=0.0,
        quality_gate=0.0,
        distinctiveness_raw=0.0,
        distinctiveness_component=0.0,
        length_penalty_component=length_penalty_component,
        language_penalty_component=language_penalty_component,
        response_n_tokens=response_n_tokens,
    )


def compute_math_reward(response: str, ground_truth: str | None = None) -> float:
    """
    Compute reward for mathematical outputs.

    Args:
        response: The generated response containing math answer
        ground_truth: The correct answer (if available in data)

    Returns:
        float: Reward score for math correctness
    """
    if isinstance(response, np.ndarray):
        response = response.tolist()[0] if response.size > 0 else ""
    response = str(response)

    if ground_truth is not None and isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()[0] if ground_truth.size > 0 else None
    if ground_truth is not None and torch.is_tensor(ground_truth):
        ground_truth = ground_truth.item()

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    hash_match = re.search(r'####\s*(.+)', response)

    predicted_answer = None
    if boxed_match:
        predicted_answer = boxed_match.group(1).strip()
    elif hash_match:
        predicted_answer = hash_match.group(1).strip()

    if predicted_answer is None:
        numbers = re.findall(r'-?\d+\.?\d*', response)
        predicted_answer = numbers[-1] if numbers else None

    if predicted_answer is None:
        return 0.0

    if ground_truth is not None and str(ground_truth).strip() != "":
        try:
            pred_num = float(predicted_answer)
            gt_num = float(ground_truth)
            if abs(pred_num - gt_num) < 1e-5:
                return 1.0
            return 0.0
        except ValueError:
            if predicted_answer.strip().lower() == ground_truth.strip().lower():
                return 1.0
            return 0.0

    return 0.5


def compute_text_quality_reward(response: str, prompt: str | None = None) -> float:
    """
    Compute reward for text quality (non-math outputs) using heuristics.

    Criteria: length, completeness, sentence count, uniqueness.
    """
    if not response or not response.strip():
        return 0.0

    reward = 0.0
    try:
        word_count = len(response.split())
        if 50 <= word_count <= 500:
            reward += 0.3
        elif 20 <= word_count < 50:
            reward += 0.15
        elif word_count > 500:
            reward += 0.1
        elif word_count == 0:
            return 0.0

        if response.strip() and response.strip()[-1] in '.!?':
            reward += 0.2

        sentence_count = len(re.split(r'[.!?]+', response))
        if sentence_count >= 3:
            reward += 0.2
        elif sentence_count >= 2:
            reward += 0.1

        words = response.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            reward += 0.3 * unique_ratio

        return min(reward, 1.0)
    except Exception as e:
        logger.error(f"Error in text quality heuristic scoring: {e}")
        return 0.1


def compute_pairwise_diversity(
    text_a: str,
    text_b: str,
    metric: str = "token_overlap",
    sbert_model=None,
    sbert_device: str | None = None,
) -> float:
    """
    Compute a diversity score between two texts in [0, 1]. Higher means more diverse.

    Supported metrics: 'token_overlap'/'jaccard', 'levenshtein'/'edit', 'sbert'.
    For 'sbert', pass sbert_model and sbert_device; if None, falls back to Jaccard.
    """
    try:
        if text_a is None:
            text_a = ""
        if text_b is None:
            text_b = ""

        a = text_a.strip()
        b = text_b.strip()

        if not a and not b:
            return 0.0
        if (not a and b) or (a and not b):
            return 1.0

        metric = (metric or "token_overlap").lower()

        if metric in ("token_overlap", "jaccard"):
            tokens_a = re.findall(r"\w+", a.lower())
            tokens_b = re.findall(r"\w+", b.lower())
            set_a = set(tokens_a)
            set_b = set(tokens_b)
            if not set_a and not set_b:
                return 0.0
            inter = set_a & set_b
            union = set_a | set_b
            jaccard = float(len(inter)) / float(len(union)) if union else 0.0
            return float(max(0.0, min(1.0, 1.0 - jaccard)))

        if metric in ("levenshtein", "edit"):
            ratio = SequenceMatcher(None, a, b).ratio()
            return float(max(0.0, min(1.0, 1.0 - ratio)))

        if metric == "sbert":
            if sbert_model is None:
                logger.warning("SBERT metric requested but model not provided. Falling back to Jaccard.")
                return compute_pairwise_diversity(a, b, metric="jaccard")
            try:
                embeddings = sbert_model.encode(
                    [a, b], convert_to_tensor=True, device=sbert_device
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                sim = float(torch.mm(embeddings[0:1], embeddings[1:2].T).item())
                sim = max(-1.0, min(1.0, sim))
                sim01 = (sim + 1.0) / 2.0
                return float(max(0.0, min(1.0, 1.0 - sim01)))
            except Exception as e:
                logger.warning(f"SBERT similarity failed: {e}. Falling back to Jaccard.")
                return compute_pairwise_diversity(a, b, metric="jaccard")

        logger.warning(f"Unknown diversity metric '{metric}', falling back to 'jaccard'.")
        return compute_pairwise_diversity(a, b, metric="jaccard")

    except Exception as e:
        logger.error(f"Error in compute_pairwise_diversity: {e}")
        return 0.0


# --- Distinctiveness / SBERT-based rewards ---
def _compute_distinctiveness_from_sims(
    other_sims: torch.Tensor,
    mode: str,
) -> float:
    """Compute distinctiveness reward from pairwise similarities. mode: 'avg' or 'min'."""
    if other_sims.numel() == 0:
        return 0.0
    if mode == "avg":
        sim = float(other_sims.mean().item())
    elif mode == "min":
        sim = float(other_sims.max().item())
    else:
        raise ValueError(f"mode must be 'avg' or 'min', got {mode}")

    sim = max(-1.0, min(1.0, sim))
    normalized_sim = (sim + 1.0) / 2.0
    distinctiveness = max(0.0, min(1.0, 1.0 - normalized_sim))
    return distinctiveness


def _compute_distinctiveness_reward_impl(
    response: str,
    all_responses_in_batch: List[str],
    mode: str,
    embeddings: torch.Tensor | None = None,
    index: int | None = None,
    encode_fn: Callable[[List[str]], torch.Tensor | None] | None = None,
    exclude_thinking: bool = False,
) -> float:
    """Shared implementation for avg/min distance rewards. Requires embeddings or encode_fn.
    When exclude_thinking is True, only the text after <think>...</think> is used for embedding distinctiveness.
    When exclude_thinking is True, encode_fn is required (pre-computed embeddings are ignored)."""
    if len(all_responses_in_batch) <= 1:
        raise ValueError("At least 2 responses are required to compute distinctiveness reward.")

    try:
        if exclude_thinking:
            response = _text_after_thinking(response)
            all_responses_in_batch = [_text_after_thinking(r) for r in all_responses_in_batch]
            embeddings = None  # Must re-encode stripped text; ignore any pre-computed embeddings

        if embeddings is None and encode_fn is not None:
            embeddings = encode_fn(all_responses_in_batch)
        if embeddings is None or embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D tensor of shape (batch_size, embedding_dim).")
        if index is None:
            try:
                index = all_responses_in_batch.index(response)
            except ValueError:
                index = 0

        sims = torch.mm(embeddings[index : index + 1], embeddings.T).squeeze(0)
        mask = torch.ones(sims.shape, dtype=torch.bool, device=sims.device)
        mask[index] = False
        other_sims = sims[mask]

        return _compute_distinctiveness_from_sims(other_sims, mode=mode)

    except Exception as e:
        logger.error(f"Error in SBERT distinctiveness scoring ({mode}): {e}.")
        raise e


def compute_avg_distance_reward(
    response: str,
    all_responses_in_batch: List[str],
    embeddings: torch.Tensor | None = None,
    index: int | None = None,
    encode_fn: Callable[[List[str]], torch.Tensor | None] | None = None,
    exclude_thinking: bool = False,
) -> float:
    """Distinctiveness reward based on average similarity to all other responses."""
    return _compute_distinctiveness_reward_impl(
        response=response,
        all_responses_in_batch=all_responses_in_batch,
        mode="avg",
        embeddings=embeddings,
        index=index,
        encode_fn=encode_fn,
        exclude_thinking=exclude_thinking,
    )

def compute_classifier_diversity_reward(
    prompt: str,
    response: str,
    all_responses_in_batch: List[str],
    embeddings: torch.Tensor | None = None,
    index: int | None = None,
    encode_fn: Callable[[List[str]], torch.Tensor | None] | None = None,
    exclude_thinking: bool = False,
    algo: str = "classifier",
) -> float:
    """Compute diversity as fraction of non-equivalent responses in the batch.

    algo: "classifier" or "gpt4" (uses equivalence checkers from EQUIVALENCE_ALGS).
    """
    if not all_responses_in_batch:
        return 0.0

    if exclude_thinking:
        response = _text_after_thinking(response)
        all_responses_in_batch = [_text_after_thinking(r) for r in all_responses_in_batch]

    algo = (algo or "classifier").lower()
    if algo == "classifier":
        equivalence_alg = equivalence_check_classifier
    elif algo == "gpt4":
        equivalence_alg = equivalence_check_gpt4
    else:
        raise ValueError(f"Unknown algo '{algo}'. Expected 'classifier' or 'gpt4'.")

    async def _count_non_equivalent() -> int:
        non_equivalent = 0
        for other in all_responses_in_batch:
            equivalent = await equivalence_alg(prompt or "", response, other)
            if not equivalent:
                non_equivalent += 1
        return non_equivalent

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_box: dict[str, int] = {}

        def _run_in_thread() -> None:
            result_box["count"] = asyncio.run(_count_non_equivalent())

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        thread.join()
        non_equivalent = result_box["count"]
    else:
        non_equivalent = asyncio.run(_count_non_equivalent())

    batch_size = len(all_responses_in_batch)
    return float(non_equivalent) / float(batch_size)
    
def compute_min_distance_reward(
    response: str,
    all_responses_in_batch: List[str],
    embeddings: torch.Tensor | None = None,
    index: int | None = None,
    encode_fn: Callable[[List[str]], torch.Tensor | None] | None = None,
    exclude_thinking: bool = False,
) -> float:
    """Distinctiveness reward based on similarity to the closest other response."""
    return _compute_distinctiveness_reward_impl(
        response=response,
        all_responses_in_batch=all_responses_in_batch,
        mode="min",
        embeddings=embeddings,
        index=index,
        encode_fn=encode_fn,
        exclude_thinking=exclude_thinking,
    )


def compute_distinctiveness_reward(
    response: str,
    all_responses_in_batch: List[str],
    embeddings: torch.Tensor | None = None,
    index: int | None = None,
    encode_fn: Callable[[List[str]], torch.Tensor | None] | None = None,
    exclude_thinking: bool = False,
    **kwargs,
) -> float:
    """
    Backward-compatible alias: same as compute_avg_distance_reward.
    Callers may pass sbert_model_name, sbert_model_device (ignored); use encode_fn for encoding.
    """
    return compute_avg_distance_reward(
        response=response,
        all_responses_in_batch=all_responses_in_batch,
        embeddings=embeddings,
        index=index,
        encode_fn=encode_fn,
        exclude_thinking=exclude_thinking,
    )


# --- G-Vendi: Gradient-based diversity (Prismatic Synthesis, arxiv.org/abs/2505.20161) ---


def compute_g_vendi_score(
    prompts: List[str],
    responses: List[str],
    proxy_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str | None = None,
    proj_dim: int = 1024,
    proj_seed: int = 42,
    max_length: int = 2048,
) -> float:
    """
    Compute G-Vendi diversity score for a set of (prompt, response) pairs.

    Uses gradient-based representation from "Prismatic Synthesis: Gradient-based
    Data Diversification Boosts Generalization in LLM Reasoning" (arxiv.org/abs/2505.20161).
    Higher score = more diverse responses (captures task-relevant diversity in gradient space).

    Args:
        prompts: Input prompts (padded with "" if shorter than responses)
        responses: Generated responses
        proxy_model_name: Small proxy model for gradient computation
        device: Device (auto if None)
        proj_dim: Rademacher projection dimension (paper: 1024)
        proj_seed: Random seed for projection
        max_length: Max sequence length

    Returns:
        G-Vendi score (>= 1.0; higher = more diverse)
    """
    from custom_reward_setup.g_vendi import compute_g_vendi_score as _g_vendi_impl

    return _g_vendi_impl(
        prompts=prompts,
        responses=responses,
        proxy_model_name=proxy_model_name,
        device=device,
        proj_dim=proj_dim,
        proj_seed=proj_seed,
        max_length=max_length,
    )
