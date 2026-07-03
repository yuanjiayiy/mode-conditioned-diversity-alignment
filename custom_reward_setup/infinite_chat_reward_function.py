# Custom reward function that handles:
# 1. Text outputs (quality assessment)
# 2. Distinctiveness/diversity scoring

import math
import re
import os
import logging
import time
import traceback
import concurrent.futures
from typing import Dict, Any, List, Callable, Optional
from collections import defaultdict
from verl.utils.nlp_utils import _text_after_thinking, compute_length_penalty, compute_language_penalty

from custom_reward_setup.reward_function_utils import (
    MATH_SOURCES,
    TEXT_SOURCES,
    SCHOLAR_SOURCES,
    DEBUG_PRINT_LIMIT,
    compute_min_distance_reward,
    compute_classifier_diversity_reward,
    resolve_device,
    to_scalar_text as _to_scalar_text,
    is_empty as _is_empty,
    normalize_seq as _normalize_seq,
    format_reward_info as _format_reward_info,
    zero_reward_info as _zero_reward_info,
    compute_math_reward,
    compute_text_quality_reward,
    compute_pairwise_diversity as _compute_pairwise_diversity_impl,
    compute_distinctiveness_reward,
)
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Whole-word "mode" — avoids matching "model", "modem", etc.
_MODE_WORD_RE = re.compile(r"\bmode\b", re.IGNORECASE)
_ROLE_WORD_RE = re.compile(r"\brole\b", re.IGNORECASE)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaForCausalLM

from verl import DataProto

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Alias for local use
_MATH_SOURCES = MATH_SOURCES
_TEXT_SOURCES = TEXT_SOURCES
_SCHOLAR_SOURCES = SCHOLAR_SOURCES
_DEBUG_PRINT_LIMIT = DEBUG_PRINT_LIMIT



# Global variables for Tulu reward model
_tulu_tokenizer = None
_tulu_model = None
_tulu_device = None

# Global variables for Prometheus reward model
_skyreward_model = None
_skyreward_device = None
_skyreward_tokenizer = None

# Global variables for SBERT model
_sbert_model = None
_sbert_device = None
_sbert_model_name = None

# 
_MATH_SOURCES = {"math", "gsm8k"}
_TEXT_SOURCES = {"text", "chat", "instruction", "liweijiang/infinite-chats-taxonomy"}
_SCHOLAR_SOURCES = {"rl-research/dr-tulu-sft-data"}

def _sigmoid(x):
    x = max(min(x, 50.0), -50.0) # avoid math range errors
    return 1.0 / (1.0 + math.exp(-x))
def _relu(x):
    return max(0.0, x)

def _is_text_source(ds: str) -> bool:
    """Return True if data source is text (exact match or ai2-adapt-dev/allenai prefix)."""
    if not ds:
        return False
    ds_lower = str(ds).lower()
    return ds_lower in _TEXT_SOURCES or ds_lower.startswith("ai2-adapt-dev/") or ds_lower.startswith("allenai/")

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())

def load_tulu_reward_model(model_name: str = "allenai/Llama-3.1-Tulu-3-8B-RM", device: str = None):
    """
    Load Tulu model for text quality assessment.
    
    Args:
        model_name: HuggingFace model name for Tulu
        device: Device to load model on (auto-detect if None)
    """
    global _tulu_tokenizer, _tulu_model, _tulu_device
    
    device = resolve_device(device)
        
    
    try:
        logger.info(f"Loading Tulu reward model: {model_name}")
        _tulu_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _tulu_tokenizer.pad_token is None:
            # simplest option: reuse EOS as PAD
            if _tulu_tokenizer.eos_token is not None:
                _tulu_tokenizer.pad_token = _tulu_tokenizer.eos_token
            else:
                # or create a new [PAD] token if EOS doesn't exist
                _tulu_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model_name="allenai/Llama-3.1-Tulu-3-8B-RM"
        _tulu_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            # num_labels=1,  # Single output for quality score
            # torch_dtype=dtype,
            # trust_remote_code=True,
        )
        print("loading the reward model on device:", device)
        _tulu_model.to(device)
        _tulu_model.eval()
        _tulu_device = device
        logger.info(f"Successfully loaded Tulu model on {device}")
    except Exception as e:
        logger.error(f"Failed to load Tulu model: {e}")
        raise


def load_skyreward_reward_model(model_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M", device: str = None):
    global _skyreward_model, _skyreward_device, _skyreward_tokenizer
    device = resolve_device(device)
    try:
        logger.info(f"Loading Skyreward reward model: {model_name} to device: {device}")
        _skyreward_device = device

        # Try flash_attention_2 first, fall back to eager if flash_attn is not installed
        for attn_impl in ("flash_attention_2", "eager"):
            try:
                _skyreward_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=_skyreward_device,
                    attn_implementation=attn_impl,
                    num_labels=1,
                )
                logger.info(f"Skyreward model loaded with attn_implementation={attn_impl}")
                break
            except Exception as attn_err:
                if attn_impl == "eager":
                    raise
                logger.warning(f"flash_attention_2 unavailable ({attn_err}), falling back to eager attention")

        _skyreward_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _skyreward_tokenizer.pad_token is None:
            if _skyreward_tokenizer.eos_token is not None:
                _skyreward_tokenizer.pad_token = _skyreward_tokenizer.eos_token
            else:
                _skyreward_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        _skyreward_model.eval()
        logger.info(f"Successfully loaded Skyreward model on {device}")
    except Exception as e:
        logger.error(f"Failed to load Skyreward model: {e}")
        raise

def load_sbert_model(model_name: str = "all-MiniLM-L6-v2", device: str = None):
    """
    Load SBERT model for semantic similarity computation.
    
    Args:
        model_name: HuggingFace model name for SBERT
        device: Device to load model on (auto-detect if None)
    """
    global _sbert_model, _sbert_device
    
    device = resolve_device(device)

    print("loading the sbert model on device: ", device)
    try:
        logger.info(f"Loading SBERT model: {model_name}")
        _sbert_model = SentenceTransformer(model_name, device=device)
        _sbert_device = device
        logger.info(f"Successfully loaded SBERT model on {device}")
        global _sbert_model_name
        _sbert_model_name = model_name
    except Exception as e:
        logger.error(f"Failed to load SBERT model: {e}")
        raise


def _encode_with_sbert(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: str | None = None,
):
    """Encode a batch of texts with SBERT and return normalized embeddings."""
    if len(texts) == 0:
        return None

    global _sbert_model, _sbert_device, _sbert_model_name
    if _sbert_model is None or (_sbert_model_name and _sbert_model_name != model_name):
        load_sbert_model(model_name=model_name, device=device)

    embeddings = _sbert_model.encode(
        texts,
        convert_to_tensor=True,
        device=_sbert_device,
        show_progress_bar=False,
    )
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, device=_sbert_device, dtype=torch.float32)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def compute_skyreward_reward(
    response: str,
    prompt: str = None,
    reference_info: dict | None = None,
    reference_score: float | None = None,
) -> float:
    """
    Compute reward using Skywork reward model (direct scoring, see skyreward.py example).
    
    Args:
        response: The generated text response
        prompt: The input prompt (optional)
    
    Returns:
        float: Quality score in [0, 1] (normalized from raw logits)
    """
    if reference_info is None and reference_score is not None:
        reference_info = {"reference_score": reference_score}
    res = compute_skyreward_reward_batch([response], [prompt or ""], reference_infos=[reference_info])
    # New API: dict with keys 'scores','gates','extras'
    if isinstance(res, dict):
        scores = res.get("scores", [])
    else:
        # Fallback to legacy tuple/list returns
        if isinstance(res, tuple) or isinstance(res, list):
            scores = res[0] if len(res) > 0 else []
        else:
            scores = []
    return scores[0] if scores else 0.0
    

def compute_reward_piecewise(raw_score, ref_mean, ref_min, ref_max):
    # Aim slightly above minimum, not merely equal to minimum.
    target_q = ref_min + 0.05 * (ref_max - ref_mean)

    tau_q = 0.25 * (ref_mean - ref_min + 1e-6)
    q_margin = (raw_score - target_q) / tau_q
    
    # q_margin can goes really large to 10k I don't know why, causing overflow in math.exp. Cap it to a reasonable range.
    # q_margin = np.clip(q_margin, -10, 10)

    quality_gate = _sigmoid(q_margin)

    # Strong below-threshold penalty.
    quality_penalty = -2.0 * _relu(-q_margin)

    # Small positive quality bonus above threshold.
    quality_bonus = 0.25 * math.tanh(q_margin)
    quality_reward = quality_penalty + quality_bonus

    return {
        "quality_reward": float(quality_reward),
        "quality_gate": float(quality_gate),
        "quality_penalty": float(quality_penalty),
        "quality_bonus": float(quality_bonus),
        "q_margin": float(q_margin),
        "tau_q": float(tau_q),
        "raw_score": float(raw_score),
        "target_q": float(target_q),
        
    }
        

def compute_reward_smooth(raw_score, ref_mean, ref_min, ref_max):
    scale = ref_max - ref_min + 1e-6
    q_norm = (raw_score - ref_mean) / scale

    temperature = 0.25

    quality_gate = _sigmoid(q_norm / temperature)
    quality_reward = math.tanh(q_norm)
    return {
        "quality_reward": float(quality_reward), 
        "quality_gate": float(quality_gate),
        "q_norm": float(q_norm),
        "temperature": float(temperature),
    }

def compute_reward_above_minimum(raw_score, ref_mean, ref_min, ref_max):
    margin = raw_score - ref_mean
    tau = 0.5 * (ref_mean - ref_min + 1e-6)

    quality_reward = np.tanh(margin / tau)
    quality_gate = np.clip(margin / tau, 0.0, 1.0)
    return {
        "quality_reward": float(quality_reward),
        "quality_gate": float(quality_gate),
        "margin": float(margin),
        "tau": float(tau),
    }

def compute_reward_naive(raw_score, ref_mean, ref_min, ref_max):
    target_q = ref_min + 0.05 * (ref_max - ref_mean)
    tau_q = 0.1 * (ref_mean - ref_min + 1e-6)
    q_margin = (raw_score - target_q) / tau_q
    quality_bonus = 0.25 * math.tanh(q_margin)
    quality_reward = quality_bonus
    quality_gate = 1.0
    return {
        "quality_reward": float(quality_reward),
        "quality_gate": float(quality_gate),
    }
    
def compute_reward_relu(raw_score, ref_mean, ref_min, ref_max):
    
    target_q = ref_min + 0.05 * (ref_max - ref_mean)
    tau_q = 0.1 * (ref_mean - ref_min + 1e-6)
    q_margin = (raw_score - target_q) / tau_q
    quality_bonus = 0.25 * math.tanh(q_margin)
    quality_reward = quality_bonus
    quality_gate = 1.0 if raw_score >= target_q else 0.0
    return {
        "quality_reward": float(quality_reward),
        "quality_gate": float(quality_gate),
    }

def compute_reward_raw(raw_score, ref_mean, ref_min, ref_max):
    return {
        "quality_reward": float(raw_score),
        "quality_gate": 1.0,
        "raw_score": float(raw_score),
    }

def compute_skyreward_reward_batch(
    responses_list: List[str],
    prompts_list: List[str] | None = None,
    reference_infos: List[dict | None] | None = None,
    max_length: int = 2048,
    micro_batch_size: int = 4,
    compute_reward: Optional[Callable[[float, Optional[float], Optional[float], Optional[float]], float]] = compute_reward_raw,
) -> List[float]:
    """
    Batched Skywork reward scoring with micro-batching to avoid OOM.
    
    Formats each (prompt, response) as a chat, tokenizes, and runs through
    the reward model in small micro-batches to fit in GPU memory alongside
    the actor/ref models.
    
    Args:
        responses_list: List of generated responses
        prompts_list: List of prompts (or None for empty prompts)
        reference_infos: List of per-sample metadata dicts. Supported keys:
            reference_score, reference_min_score, reference_max_score.
            Ignored when reference_infos is provided.
        max_length: Max token length per sequence
        micro_batch_size: Number of items to score at once (lower = less memory)
    
    Returns:
        List[float]: One score per item
        List[float]: One gate value per item (if compute_reward returns gate)
    """
    global _skyreward_model, _skyreward_tokenizer, _skyreward_device
    
    if _skyreward_model is None or _skyreward_tokenizer is None:
        try:
            initialize_skyreward_model()
        except Exception as e:
            logger.warning(f"Failed to initialize Skyreward model: {e}. Returning zeros.")
            return {"scores": [0.0 for _ in responses_list], "gates": [0.0 for _ in responses_list], "extras": [{} for _ in responses_list]}
    
    prompts_list = prompts_list or [""] * len(responses_list)
    
    # Remove thinking tokens
    responses_list = [_text_after_thinking(response) for response in responses_list]
    
    n = len(responses_list)
    if n == 0:
        return {"scores": [], "gates": [], "extras": []}

    start = time.time()
    
    
    formatted_texts = []
    for prompt, response in zip(prompts_list, responses_list):
        conv = [{"role": "user", "content": prompt or ""}, {"role": "assistant", "content": response or ""}]
        formatted = _skyreward_tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        if _skyreward_tokenizer.bos_token is not None and formatted.startswith(_skyreward_tokenizer.bos_token):
            formatted = formatted[len(_skyreward_tokenizer.bos_token):]
        formatted_texts.append(formatted)
    
    raw_scores = []
    for mb_start in range(0, n, micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, n)
        mb_texts = formatted_texts[mb_start:mb_end]
        
        batch = _skyreward_tokenizer(
            mb_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        batch = {k: v.to(_skyreward_model.device) for k, v in batch.items()}
        
        with torch.no_grad():
            logits = _skyreward_model(**batch).logits
        mb_scores = logits.squeeze(-1).cpu().float().tolist()
        if isinstance(mb_scores, float):
            mb_scores = [mb_scores]
        raw_scores.extend(mb_scores)
        
        del batch, logits
        torch.cuda.empty_cache()

    if reference_infos is not None:
        assert len(raw_scores) == len(reference_infos), "Number of raw scores and reference infos must match"
        scores = []
        gates = []
        extras = []
        for s, info in zip(raw_scores, reference_infos):
            if not info:
                # No reference info: for base models use compute_reward_raw
                ret = compute_reward_raw(s, None, None, None)
                if isinstance(ret, dict):
                    q = float(ret.get("quality_reward", 0.0))
                    g = float(ret.get("quality_gate", 0.0))
                    extra = {k: v for k, v in ret.items() if k not in ("quality_reward", "quality_gate")}
                else:
                    q, g = ret
                    extra = {}
                scores.append(float(q))
                gates.append(float(g))
                extras.append(extra)
                continue

            ref_mean = info.get("reference_mean_score", None)
            ref_min = info.get("reference_min_score", None)
            ref_max = info.get("reference_max_score", None)

            # If user provided a compute_reward function, defer to it.
            try:
                ret = compute_reward(s, ref_mean, ref_min, ref_max)
                if isinstance(ret, dict):
                    q = float(ret.get("quality_reward", 0.0))
                    g = float(ret.get("quality_gate", 0.0))
                    extra = {k: v for k, v in ret.items() if k not in ("quality_reward", "quality_gate")}
                else:
                    q, g = ret
                    extra = {}
                scores.append(float(q))
                gates.append(float(g))
                extras.append(extra)
                continue
            except Exception as e:
                logger.warning(f"compute_reward callback raised an exception: {e}; falling back to default behavior")
                raise e

    end = time.time()
    logger.info(f"Skyreward batch ({n} items, micro_batch={micro_batch_size}) took {end - start:.2f}s")
    return {"scores": scores, "gates": gates, "extras": extras}


def compute_pairwise_diversity(text_a: str, text_b: str, metric: str = "token_overlap") -> float:
    """Wrapper that passes module's SBERT model to utils."""
    return _compute_pairwise_diversity_impl(
        text_a, text_b, metric=metric,
        sbert_model=_sbert_model,
        sbert_device=_sbert_device,
    )

    
def initialize_skyreward_model(model_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M", device: str = None):
    """
    Initialize the Skyreward reward model. Call this before using the reward function.
    
    Args:
        model_name: HuggingFace model name for Skyreward
        device: Device to load model on (auto-detect if None)
    """
    print("Initializing Skyreward reward model...")
    load_skyreward_reward_model(model_name, device)


def combine_reward_components(
    base_reward: float,
    quality_gate: float = 1.0,
    distinct_component: float = 0.0,
    length_penalty_component: float = 0.0,
    language_penalty_component: float = 0.0,
    format_penalty_component: float = 0.0,
    text_quality_alpha: float = 1.0,
    mode: str = "item",
) -> float:
    """Combine reward components into a final scalar reward.

    Parameters
    - base_reward: the primary quality score (in [0,1])
    - quality_gate: a gating value (in [0,1]) that can be used to modulate the influence of base_reward in batch mode
    - distinct_component: already-scaled distinctiveness component
    - length_penalty_component: penalty (can be negative) for long responses
    - language_penalty_component: penalty for language issues
    - format_penalty_component: penalty for format violations
    - text_quality_alpha: multiplier for base_reward in batch mode
    - mode: either "item" (NaiveRewardManager) or "batch" (BatchRewardManager)

    Returns
    - final_reward: float

    Behavior notes:
    - "item" mode preserves previous single-item contract: clip(quality+distinct) then add penalties.
    - "batch" mode uses linear combination: distinct + alpha * quality + penalties.
    """
    if mode == "item":
        clipped = max(0.0, min(1.0, float(base_reward) + float(distinct_component or 0.0)))
        final = clipped + float(language_penalty_component or 0.0) + float(format_penalty_component or 0.0)
        return float(final)

    # batch mode (default fallback)
    final = base_reward + quality_gate * distinct_component
    final += float(length_penalty_component or 0.0) + float(language_penalty_component or 0.0) + float(format_penalty_component or 0.0)
    return float(final)

def hybrid_reward_function_item(
    data_source: str,
    solution_str: str,
    ground_truth: str = None,
    extra_info: dict = None,
    math_weight: float = 1.0,
    text_weight: float = 1.0,
    distinctiveness_weight: float = 0.3,
    skyreward_model_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
    sbert_model_name: str = "all-MiniLM-L6-v2",
    skyreward_model_device: str | None = None,
    sbert_model_device: str | None = None,
    length_penalty_weight: float = 0.0,
    length_penalty_max_tokens: int = 512,
    language_penalty_weight: float = 0.0,
    format_penalty_weight: float = 0.0,
    **kwargs
) -> float:
    """
    Process a single item reward (called by NaiveRewardManager).
    
    Args:
        data_source: Type of task (e.g., 'math', 'text', 'chat', 'instruction')
        solution_str: The generated response
        ground_truth: Ground truth answer (if available)
        extra_info: Additional information dict (may contain 'prompt' or batch context)
        math_weight: Weight for math correctness reward
        text_weight: Weight for text quality reward
        distinctiveness_weight: Weight for distinctiveness reward (not used in single-item mode)
        tulu_model_name: HuggingFace model name for Tulu reward model
        sbert_model_name: HuggingFace model name for SBERT distinctiveness model
        **kwargs: Additional keyword arguments
    
    Returns:
        float: Reward score for this item
    """
    try:
        solution_str = _to_scalar_text(solution_str)
        ground_truth = _to_scalar_text(ground_truth)

        if _is_empty(solution_str):
            logger.warning("Empty solution_str provided, returning 0.0")
            return 0.0
        
        # Initialize models if needed
        global _skyreward_model
        
        # Extract prompt from extra_info if available
        prompt = None
        if len(extra_info) > 0:
            prompt = _to_scalar_text(extra_info.get("prompt") or extra_info.get("question"))
        
        # Initialize reward components
        math_reward = 0.0
        base_reward = 0.0
        
        # Normalize data_source to lowercase for comparison
        data_source_lower = str(data_source).lower() if data_source else 'text'
        
        math_component = 0.0
        text_component = 0.0
        base_reward = 0.0

        # Determine task type and compute appropriate rewards
        if data_source_lower in _MATH_SOURCES:
            math_reward = compute_math_reward(solution_str, ground_truth)
            math_component = math_weight * math_reward
            base_reward = math_component

        elif _is_text_source(data_source_lower):
            if _skyreward_model is None:
                try:
                    initialize_skyreward_model(skyreward_model_name, device=skyreward_model_device)
                except Exception as e:
                    logger.warning(f"Failed to initialize Skyreward model: {e}.")
            reference_score = extra_info.get("reference_score", None)
            base_reward = compute_skyreward_reward(solution_str, prompt, reference_score=reference_score)
            text_component = text_weight * base_reward
            base_reward = text_component
        
        elif data_source_lower in _SCHOLAR_SOURCES:                    
            base_reward = compute_skyreward_reward(solution_str, prompt)
            text_component = text_weight * base_reward
            base_reward = text_component

        else:
            raise ValueError(f"Invalid data source: {data_source_lower}")

        base_reward = float(base_reward)
        base_reward = max(0.0, min(1.0, base_reward))

        # No batch context for distinctiveness in single-item mode
        distinct_reward = 0.0
        distinct_component = 0.0
        language_penalty_component = 0.0
        if language_penalty_weight > 0.0:
            language_penalty_component, _, _ = compute_language_penalty(
                prompt=prompt or "",
                response=solution_str,
                weight=language_penalty_weight,
            )
        format_penalty_component = 0.0
        if format_penalty_weight > 0.0:
            format_penalty_component = -1.0 if _ROLE_WORD_RE.search(solution_str) else 0.0
            format_penalty_component = format_penalty_component * format_penalty_weight

            
        final_reward = combine_reward_components(
            base_reward=base_reward,
            distinct_component=distinct_component,
            length_penalty_component=0.0,
            language_penalty_component=language_penalty_component,
            format_penalty_component=format_penalty_component,
            mode="item",
        )

        return _format_reward_info(
            total_reward=final_reward,
            base_reward=base_reward,
            math_component=math_component,
            text_component=text_component,
            math_raw=math_reward,
            text_raw=base_reward,
            distinctiveness_component=distinct_reward,
            language_penalty_component=language_penalty_component,
            format_reward=format_penalty_component,
        )

    except Exception as e:
        logger.error(f"Error in hybrid_reward_function_item: {e}")
        logger.error(traceback.format_exc())
        # Return minimal reward on error to avoid breaking training
        return 0.0


def hybrid_reward_function_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str] = None,
    extra_infos: List[dict] = None,
    math_weight: float = 1.0,
    text_weight: float = 1.0,
    distinctiveness_weight: float = 0.3,
    skyreward_model_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
    sbert_model_name: str = "all-MiniLM-L6-v2",
    skyreward_model_device: str | None = None,
    sbert_model_device: str | None = None,
    thresholding_base_reward: float | None = 1.0,
    length_penalty_weight: float = 0.0,
    length_penalty_max_tokens: int = 512,
    language_penalty_weight: float = 0.0,
    format_penalty_weight: float = 0.0,
    text_quality_alpha: float = 1.0,
    compute_distinct: Callable[..., float] = "compute_min_distance_reward",
    distinct_classifier_algorithm: str = "classifier",
    **kwargs
) -> List[float]:
    """
    Process batch rewards (called by BatchRewardManager).
    
    Args:
        data_sources: List of task types for each item
        solution_strs: List of generated responses
        ground_truths: List of ground truth answers (if available)
        extra_infos: List of additional information dicts (may contain 'prompt')
        math_weight: Weight for math correctness reward
        text_weight: Weight for text quality reward
        distinctiveness_weight: Weight for distinctiveness reward
        text_quality_alpha: Scaling factor for base_reward in final combination
            (final = distinct_reward + text_quality_alpha * base_reward + penalties)
        tulu_model_name: HuggingFace model name for Tulu reward model
        sbert_model_name: HuggingFace model name for SBERT distinctiveness model
        **kwargs: Additional keyword arguments
    
    Returns:
        List[dict]: One dict per item with breakdown info and final score
    """
    try:
        # Validate inputs
        if len(solution_strs) == 0:
            logger.warning("Empty solution_strs provided, returning empty list")
            return []
        
        
        # Initialize models if needed
        global _skyreward_model
        
        if _sbert_model is None:
            try:
                load_sbert_model(sbert_model_name, device=sbert_model_device)
            except Exception as e:
                logger.warning(f"Failed to initialize SBERT model: {e}")
        
        batch_size = len(solution_strs)
        rewards = []

        cleaned_responses = [_to_scalar_text(s) for s in solution_strs]
        
        # Precompute SBERT embeddings for all responses if distinctiveness is needed
        sbert_embeddings = None
        if distinctiveness_weight > 0 and batch_size > 1:
            try:
                sbert_embeddings = _encode_with_sbert(
                    cleaned_responses,
                    model_name=sbert_model_name,
                    device=sbert_model_device,
                )
            except Exception as e:
                logger.warning(f"Failed to precompute SBERT embeddings: {e}")
                sbert_embeddings = None

        # Precompute prompts for batch (if provided)
        batch_prompts: List[str] = []
        reference_infos: List[dict] = []
        extra_infos = extra_infos if extra_infos is not None else []
        if len(extra_infos) > 0:
            for i in range(batch_size):
                info = extra_infos[i] if i < len(extra_infos) else None
                p = _to_scalar_text(info.get('prompt') or info.get('question', None)) if info else ""
                batch_prompts.append(p)
                reference_infos.append(info if isinstance(info, dict) else {})
        else:
            batch_prompts = [""] * batch_size
        
        # Precompute Skyreward scores for TEXT/SCHOLAR items (batched)
        skyreward_by_idx: Dict[int, float] = {}
        skyreward_indices = [
            i for i in range(batch_size)
            if not _is_empty(cleaned_responses[i])
            and _is_text_source(str(data_sources[i] if i < len(data_sources) else "text"))
        ]
        if skyreward_indices:
            if _skyreward_model is None:
                try:
                    initialize_skyreward_model(skyreward_model_name, device=skyreward_model_device)
                except Exception as e:
                    logger.warning(f"Failed to initialize Skyreward model: {e}")
            if _skyreward_model is not None:
                try:
                    resp_subset = [cleaned_responses[i] for i in skyreward_indices]
                    prompt_subset = [batch_prompts[i] for i in skyreward_indices]
                    reference_info_subset = [reference_infos[i] for i in skyreward_indices] if reference_infos else None
                    # Resolve optional compute_reward callable from kwargs (can be a name string or a callable)
                    compute_reward_fn = kwargs.get("compute_reward", None)
                    print("compute_reward_fn from kwargs:", compute_reward_fn, "compute_distinct:", compute_distinct, "distinct_classifier_algorithm:", distinct_classifier_algorithm)
                    if isinstance(compute_reward_fn, str):
                        compute_reward_fn = globals().get(compute_reward_fn, None)
                    if compute_reward_fn is None:
                        compute_reward_fn = compute_reward_raw

                    res = compute_skyreward_reward_batch(
                        resp_subset,
                        prompt_subset,
                        reference_infos=reference_info_subset,
                        compute_reward=compute_reward_fn,
                    )
                    skyreward_scores = res.get("scores", [])
                    skyreward_gates = res.get("gates", [])
                    skyreward_extras = res.get("extras", [{} for _ in skyreward_scores])

                    skyreward_by_idx = {
                        skyreward_indices[j]: (skyreward_scores[j], skyreward_gates[j], skyreward_extras[j])
                        for j in range(len(skyreward_indices))
                    }
                except Exception as e:
                    logger.warning(f"Skyreward batch failed: {e}")
                    traceback.print_exc()
                    raise Exception("Skyreward batch scoring failed") from e
        
        
        # Build mapping from prompt to indices (once)
        # Note: small overhead to construct per item; keeps patch minimal
        prompt_to_indices = {}
        for idx, p in enumerate(batch_prompts):
            prompt_to_indices.setdefault(p, []).append(idx)
        
        debug_printed = 0
        distinct_by_index: Dict[int, float] = {}
        compute_distinct_fn = None
        if callable(compute_distinct):
            compute_distinct_fn = compute_distinct
        elif isinstance(compute_distinct, str):
            compute_distinct_fn = globals().get(compute_distinct, None)

        if distinctiveness_weight > 0 and batch_size > 1 and compute_distinct_fn is not None:
            tasks: List[tuple[int, concurrent.futures.Future]] = []
            max_workers = min(16, len(cleaned_responses))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for prompt_key, group_indices in prompt_to_indices.items():
                    if len(group_indices) <= 1:
                        continue
                    group_embeddings = (
                        sbert_embeddings[group_indices] if sbert_embeddings is not None else None
                    )
                    group_cleaned_responses = [cleaned_responses[j] for j in group_indices]
                    for pos, idx in enumerate(group_indices):
                        solution_str = group_cleaned_responses[pos]
                        if compute_distinct_fn == compute_classifier_diversity_reward:
                            future = executor.submit(
                                compute_distinct_fn,
                                batch_prompts[idx] if idx < len(batch_prompts) else "",
                                solution_str,
                                group_cleaned_responses,
                                exclude_thinking=True,
                                algo=distinct_classifier_algorithm,
                            )
                        else:
                            future = executor.submit(
                                compute_distinct_fn,
                                solution_str,
                                group_cleaned_responses,
                                embeddings=group_embeddings,
                                index=pos,
                                encode_fn=lambda texts: _encode_with_sbert(
                                    texts, model_name=sbert_model_name, device=sbert_model_device
                                ),
                                exclude_thinking=True,
                            )
                        tasks.append((idx, future))

                future_to_idx = {future: idx for idx, future in tasks}
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc="distinct_reward",
                    leave=False,
                ):
                    idx = future_to_idx[future]
                    try:
                        distinct_by_index[idx] = float(future.result())
                    except Exception as e:
                        logger.warning(f"Distinctiveness computation failed for idx {idx}: {e}")
                        distinct_by_index[idx] = 0.0

        # Process each item in the batch
        for i in range(batch_size):
            try:
                data_source_raw = data_sources[i] if i < len(data_sources) else 'text'
                data_source = _to_scalar_text(data_source_raw)
                solution_str_raw = cleaned_responses[i] if i < len(cleaned_responses) else ''
                solution_str = _to_scalar_text(solution_str_raw)

                if _is_empty(solution_str):
                    logger.warning(f"Empty solution_str at index {i}, assigning reward 0.0")
                    rewards.append(_zero_reward_info())
                    continue

                prompt = batch_prompts[i] if i < len(batch_prompts) else ""

                math_reward = 0.0
                entry = skyreward_by_idx.get(i, (0.0, 0.0, {}))
                # entry can be (score, gate, extras)
                if isinstance(entry, tuple) or isinstance(entry, list):
                    quality_reward = float(entry[0])
                    quality_gate = float(entry[1])
                    sky_extras = entry[2] if len(entry) > 2 else {}
                else:
                    quality_reward = float(entry)
                    quality_gate = 0.0
                    sky_extras = {}
                
                data_source_lower = str(data_source).lower() if not _is_empty(str(data_source)) else 'text'

                if data_source_lower in _MATH_SOURCES:
                    math_reward = compute_math_reward(solution_str, ground_truths[i] if ground_truths and i < len(ground_truths) else None)

                math_component = math_weight * math_reward
                text_component = text_weight * quality_reward
                base_reward = text_component

                distinct_reward = distinct_by_index.get(i, 0.0)
                distinct_component = distinctiveness_weight * distinct_reward
                length_penalty_component = 0.0
                n_tokens_dict = {}
                if length_penalty_weight > 0.0:
                    length_penalty_component, n_tokens_dict = compute_length_penalty(
                            response=solution_str,
                            tokenizer=_tulu_tokenizer,
                            max_tokens=length_penalty_max_tokens,
                            weight=length_penalty_weight,
                            exclude_thinking=True,
                    )
                n_tokens = n_tokens_dict.get("response_n_tokens", 0)
                thinking_n_tokens = n_tokens_dict.get("thinking_n_tokens", 0)
                answer_n_tokens = n_tokens_dict.get("answer_n_tokens", 0)
                
                language_penalty_component = 0.0
                if language_penalty_weight > 0.0:
                    language_penalty_component, _, _ = compute_language_penalty(
                        prompt=prompt,
                        response=solution_str,
                        weight=language_penalty_weight,
                        exclude_thinking=True,
                    )
                    
                format_penalty_component = 0.0
                if format_penalty_weight > 0.0:
                    format_penalty_component = -1.0 if _ROLE_WORD_RE.search(solution_str) else 0.0
                    format_penalty_component = format_penalty_component * format_penalty_weight

                # reward shaping to get quality signal to 1
                final_reward = combine_reward_components(
                    base_reward=base_reward,
                    quality_gate=quality_gate,
                    distinct_component=distinct_component,
                    length_penalty_component=length_penalty_component,
                    language_penalty_component=language_penalty_component,
                    format_penalty_component=format_penalty_component,
                    text_quality_alpha=text_quality_alpha,
                    mode="batch",
                )

                # merge extra skyreward metadata into reward_info arguments
                reward_info = _format_reward_info(
                    total_reward=final_reward,
                    base_reward=base_reward,
                    math_component=math_component,
                    text_component=text_component,
                    math_raw=math_reward,
                    text_raw=quality_reward,
                    quality_gate=quality_gate,
                    distinctiveness_raw=distinct_reward,
                    distinctiveness_component=distinct_component,
                    length_penalty_component=length_penalty_component,
                    language_penalty_component=language_penalty_component,
                    response_n_tokens=n_tokens,
                    thinking_n_tokens=thinking_n_tokens,
                    answer_n_tokens=answer_n_tokens,
                    format_reward=format_penalty_component,
                    **(sky_extras or {}))
                rewards.append(reward_info)

                if _DEBUG_PRINT_LIMIT > 0 and debug_printed < _DEBUG_PRINT_LIMIT:
                    logger.info(
                        f"[reward_debug] idx={i} ds={data_source_lower} "
                        f"math={math_reward:.3f} text={base_reward:.3f} distinct={distinct_reward:.3f} "
                        f"length_penalty={length_penalty_component:.3f} lang_penalty={language_penalty_component:.3f} "
                        f"format={format_penalty_component:.1f} total={final_reward:.3f}"
                    )
                    logger.info(f"[reward_debug] prompt: {prompt[:200] if prompt else ''}")
                    logger.info(f"[reward_debug] response: {solution_str[:200]}")
                    debug_printed += 1

            except Exception as e:
                logger.error(f"Error processing batch item {i}: {e}")
                logger.error(traceback.format_exc())
                rewards.append(_zero_reward_info())

        # Log the best-scoring response for quick inspection
        if _DEBUG_PRINT_LIMIT > 0 and rewards:
            scores = [r["score"] for r in rewards]
            best_idx = int(np.argmax(scores))
            logger.info(
                f"[reward_debug] best idx={best_idx} score={scores[best_idx]:.3f} "
                f"math={rewards[best_idx]['accuracy_reward']:.3f} "
                f"text={rewards[best_idx]['base_reward']:.3f} "
                f"distinct={rewards[best_idx]['distinctiveness_reward']:.3f} "
                f"length_penalty={rewards[best_idx]['length_penalty']:.3f} "
                f"format={rewards[best_idx].get('format_reward', 1.0):.1f}"
            )
            logger.info(f"[reward_debug] best response: {_to_scalar_text(cleaned_responses[best_idx])[:400]}")

        return rewards
        
    except Exception as e:
        logger.error(f"Critical error in hybrid_reward_function_batch: {e}")
        logger.error(traceback.format_exc())
        count = len(solution_strs) if solution_strs else 0
        return [_zero_reward_info() for _ in range(count)]


def hybrid_reward_function(
    *args,
    math_weight: float = 1.0,
    text_weight: float = 1.0,
    distinctiveness_weight: float = 0.3,
    tulu_model_name: str = "allenai/tulu-2-7b",
    sbert_model_name: str = "all-MiniLM-L6-v2",
    tulu_model_device: str | None = None,
    sbert_model_device: str | None = None,
    compute_reward: Optional[Callable[[float, Optional[float], Optional[float], Optional[float]], float]] = compute_reward_raw,
    compute_distinct: Callable[..., float] = compute_min_distance_reward,
    distinct_classifier_algorithm: str = "classifier",
    **kwargs
):
    """
    Main custom reward function that handles both math and text outputs
    with distinctiveness scoring. This function detects the input signature
    and routes to the appropriate handler.
    
    Supported signatures:
    1. NaiveRewardManager: (data_source=..., solution_str=..., ground_truth=..., extra_info=...)
    2. BatchRewardManager: (data_sources=..., solution_strs=..., ground_truths=..., extra_infos=...)
    
    Args:
        *args: Variable positional arguments (legacy support)
        math_weight: Weight for math correctness reward
        text_weight: Weight for text quality reward
        distinctiveness_weight: Weight for distinctiveness reward
        tulu_model_name: HuggingFace model name for Tulu reward model
        sbert_model_name: HuggingFace model name for SBERT distinctiveness model
        **kwargs: Keyword arguments including data_source/solution_str (NaiveRewardManager)
                 or data_sources/solution_strs (BatchRewardManager)
    
    Returns:
        float: Single reward score (for NaiveRewardManager)
        OR
        List[float]: List of reward scores (for BatchRewardManager)
        OR
        dict: Dictionary with 'reward_tensor' and 'reward_extra_info' (for DataProto mode)
    """
    # Check keyword arguments first (most common for reward managers)
    data_source = kwargs.get('data_source', None)
    solution_str = kwargs.get('solution_str', None)
    data_sources = kwargs.get('data_sources', None)
    solution_strs = kwargs.get('solution_strs', None)
    
    # BatchRewardManager mode - check for batch keyword args
    if data_sources is not None and solution_strs is not None:
        print("Use of batch mode in hybrid_reward_function")
        # Batch mode via kwargs
        return hybrid_reward_function_batch(
            data_sources=data_sources,
            solution_strs=solution_strs,
            ground_truths=kwargs.get('ground_truths', None),
            extra_infos=kwargs.get('extra_infos', None),
            math_weight=math_weight,
            text_weight=text_weight,
            distinctiveness_weight=distinctiveness_weight,
            tulu_model_name=tulu_model_name,
            sbert_model_name=sbert_model_name,
            tulu_model_device=tulu_model_device,
            sbert_model_device=sbert_model_device,
            compute_reward=compute_reward,
            compute_distinct=compute_distinct,
            distinct_classifier_algorithm=distinct_classifier_algorithm,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    'data_sources',
                    'solution_strs',
                    'ground_truths',
                    'extra_infos',
                    'math_weight',
                    'text_weight',
                    'distinctiveness_weight',
                    'compute_reward',
                    'compute_distinct',
                    'distinct_classifier_algorithm',
                    'tulu_model_name',
                    'sbert_model_name',
                    'tulu_model_device',
                    'sbert_model_device',
                ]
            }
        )
    
    # NaiveRewardManager mode - check for single-item keyword args
    if data_source is not None and solution_str is not None:
        print("Use of item mode in hybrid_reward_function")
        # Single-item mode via kwargs
        return hybrid_reward_function_item(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=kwargs.get('ground_truth', None),
            extra_info=kwargs.get('extra_info', None),
            math_weight=math_weight,
            text_weight=text_weight,
            distinctiveness_weight=distinctiveness_weight,
            tulu_model_name=tulu_model_name,
            sbert_model_name=sbert_model_name,
            tulu_model_device=tulu_model_device,
            sbert_model_device=sbert_model_device,
            compute_reward=compute_reward,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    'data_source',
                    'solution_str',
                    'ground_truth',
                    'extra_info',
                    'math_weight',
                    'text_weight',
                    'distinctiveness_weight',
                    'tulu_model_name',
                    'sbert_model_name',
                    'tulu_model_device',
                    'sbert_model_device',
                    'compute_reward',
                ]
            }
        )
    
    # If we get here, we couldn't detect the signature
    raise ValueError(
        f"Unable to detect input signature. "
        f"Expected either (data_source=..., solution_str=...) for NaiveRewardManager "
        f"or (data_sources=..., solution_strs=...) for BatchRewardManager. "
        f"Got args: {args}, kwargs keys: {list(kwargs.keys())}"
    )


# Alternative: If you want just the tensor (backward compatible)
def hybrid_reward_function_simple(*args, **kwargs):
    """
    Simplified version that returns only the reward tensor or score.
    """
    result = hybrid_reward_function(*args, **kwargs)
    if isinstance(result, dict):
        return result.get('reward_tensor', result)
    return result
