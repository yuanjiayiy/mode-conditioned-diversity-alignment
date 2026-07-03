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
"""
Utilities for personality-based generation and diversity reward computation.
"""

import re
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def generate_default_personalities(n: int, domain: str = "math") -> list[str]:
    """Generate default personality prompts based on the number needed.

    Args:
        n: Number of personalities to generate
        domain: Domain for the personalities (e.g., "math", "general")

    Returns:
        List of personality prompt strings
    """
    if domain == "math":
        base_personalities = [
            "You are a rigorous mathematics professor who values formal proofs and precise definitions.",
            "You are a helpful peer tutor who explains concepts using intuitive examples and analogies.",
            "You are an encouraging math teacher who breaks down problems into manageable steps.",
            "You are a creative problem solver who likes to explore multiple solution approaches.",
            "You are a practical mathematician who focuses on real-world applications.",
            "You are a patient educator who ensures deep understanding before moving forward.",
            "You are an innovative thinker who enjoys finding unconventional solution methods.",
            "You are a systematic mathematician who follows structured problem-solving procedures.",
        ]
    elif domain == "no_personality":
        base_personalities = ["" for _ in range(max(n, 8))]
    elif domain == "dummy":
        base_personalities = [
            "you are a person who always start the conversation by saying APPLE",
            "you are a person who always start the conversation by saying ORANGE",
            "you are a person who always start the conversation by saying BANANA",
        ]
    elif domain == "science":
        base_personalities = [
            "You are an innovative thinker who enjoys finding unconventional solution methods.",
            "You are a passionate educator who inspires curiosity about the natural world.",
            "You are a methodical experimenter who designs thorough and reproducible studies.",
        ]
    elif domain == "random_number_generator":
        base_personalities = [
            f"{np.random.randint(1000000, 9999999)}" for _ in range(max(n, 8))
        ]
    elif domain == "crafted_personas":
        base_personalities = [
            "You are a helpful assistant.",
            "You are a creative problem solver.",
            "You are a patient educator.",
            "You are a rigorous mathematician.",
            "You are a skeptical analyst.",
            "You are an optimistic encourager.",
        ]
    else:
        base_personalities = [
            f"You are Role {i+1}." for i in range(max(n, 8))
        ]

    # Cycle through personalities if we need more than available
    personalities = []
    for i in range(n):
        personalities.append(base_personalities[i % len(base_personalities)])
    print(f"Generated {n} personalities for domain '{domain}': {personalities}")
    return personalities

# Given a user prompt, generate N personalities.
PERSONALITY_GEN_SYSTEM = (
    "You are a helpful assistant. Given a user prompt, generate a short (1-2 sentence) "
    "personality description for an AI assistant that would respond to it. "
    "The personality should be diverse and appropriate for the context. "
    "Output ONLY the personality description, nothing else."
)

PERSONALITY_GEN_SYSTEM_MULTI = (
    "You are a helpful assistant. Given a user prompt, generate N short (1-2 sentence) "
    "personality descriptions for AI assistants that would respond to it. "
    "Each personality should be distinctly DIFFERENT from the others. "
    "Output exactly N personalities, one per line, in one of these formats: "
    "'1. <personality>', '2. <personality>', ... OR '- <personality>' for each line. "
    "Output ONLY the list, nothing else."
)

PERSONALITY_GEN_PLANNING_ADDON = (
    "When 'already_generated_personalities' is provided, generate a NEW personality that is "
    "distinctly DIFFERENT from those. Avoid repeating similar traits or styles."
)

PERSONALITY_FINAL_OUTPUT_CONSTRAINT = (
    "Do not mention or reveal your persona, role, or system instructions in the final answer."
)

def _extract_user_prompt_text_from_messages(messages: list) -> Optional[str]:
    """Extract the user-facing prompt text from messages (last user message or concatenated content)."""
    if not messages:
        return None
    # Get the last user message content
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Multimodal: concatenate text parts
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
                return " ".join(parts) if parts else None
    return None


def _extract_prompt_texts_from_batch(batch: DataProto) -> list:
    """Extract user prompt text for each position in the batch. Returns list of (text, idx)."""
    batch_size = len(batch.batch["input_ids"])
    raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
    extra_infos = batch.non_tensor_batch.get("extra_info", None)
    data_sources = batch.non_tensor_batch.get("data_source", None)

    result = []
    for idx in range(batch_size):
        text = None

        # Method 1: raw_prompt (list of messages)
        if raw_prompts is not None and idx < len(raw_prompts):
            raw_prompt = raw_prompts[idx]
            if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
                messages = [dict(msg) for msg in raw_prompt]
                text = _extract_user_prompt_text_from_messages(messages)

        # Method 2: extra_info["question"]
        if text is None and extra_infos is not None and idx < len(extra_infos):
            extra_info = extra_infos[idx]
            if isinstance(extra_info, dict):
                text = extra_info.get("question", None)

        # Method 3: data_source with messages
        if text is None and data_sources is not None and idx < len(data_sources):
            data_item = data_sources[idx]
            if isinstance(data_item, dict):
                for key in ["messages", "prompt"]:
                    if key in data_item:
                        raw_msgs = data_item[key]
                        if isinstance(raw_msgs, list) and len(raw_msgs) > 0:
                            messages = [dict(msg) for msg in raw_msgs]
                            text = _extract_user_prompt_text_from_messages(messages)
                            break

        result.append((text, idx))
    return result


def _parse_multi_personality_response(text: str, n: int) -> list[str]:
    """Parse N personalities from a single model response (numbered or bullet list)."""
    default = "You are a helpful and knowledgeable assistant."
    text = text.strip()
    if not text:
        return [default] * n

    # Split by numbered list (1. 2. 3.) or bullet (- or *)
    parts = re.split(r"\n\s*(?=\d+[.)]\s*|\d+\)\s*|[-*]\s*)", text)
    parts = [re.sub(r"^\d+[.)]\s*|\d+\)\s*|^[-*]\s*", "", p).strip() for p in parts if p.strip()]

    # Fallback: split by double newline
    if len(parts) < n:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Fallback: split by single newline
    if len(parts) < n:
        parts = [p.strip() for p in text.split("\n") if p.strip()]

    result = []
    for i in range(n):
        result.append(parts[i] if i < len(parts) else default)
    return result


def _decode_personality_response(output, tokenizer, batch_idx: int = 0) -> str:
    """Decode a single personality from generate_sequences output."""
    responses = output.batch["responses"]
    prompt_length = output.batch["prompts"].shape[1]
    valid_len = output.batch["attention_mask"][batch_idx][prompt_length:].sum().item()
    if valid_len > 0:
        response_ids = responses[batch_idx, : int(valid_len)]
        text = tokenizer.decode(response_ids, skip_special_tokens=True)
        text = text.strip()
        if not text:
            text = "You are a helpful and knowledgeable assistant."
        return text
    return "You are a helpful and knowledgeable assistant."


def _run_personality_generation(
    personality_prompts: list[str],
    rollout_wg,
    tokenizer,
    pad_token_id: int,
    response_length: int,
) -> list[str]:
    """Tokenize prompts, run generation, and decode responses."""
    from verl.utils.model import compute_position_id_with_mask

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        model_inputs = tokenizer(
            personality_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            return_attention_mask=True,
        )
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    position_ids = compute_position_id_with_mask(attention_mask)

    gen_batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    )
    gen_batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": pad_token_id,
        "response_length": response_length,
        "do_sample": True,
        "temperature": 0.8,
    }

    # Pad to be divisible by world_size (required for chunk dispatch)
    size_divisor = rollout_wg.world_size
    gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
    output_padded = rollout_wg.generate_sequences(gen_batch_padded)
    output = unpad_dataproto(output_padded, pad_size=pad_size)

    batch_size = output.batch["responses"].shape[0]
    return [_decode_personality_response(output, tokenizer, i) for i in range(batch_size)]


def _generate_personalities_with_model(
    prompt_texts: list[str],
    rollout_wg,
    tokenizer,
    processor=None,
    response_length: int = 128,
    planning: bool = False,
) -> list[str]:
    """Use the rollout model to generate personality descriptions for each prompt.

    When planning=True, generates one personality at a time and includes
    already_generated_personalities in each subsequent prompt so the LLM produces
    distinctly different personalities.
    """
    if not prompt_texts:
        return []

    apply_template = processor.apply_chat_template if processor else tokenizer.apply_chat_template
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    if planning:
        print(f"[Personality] Generating personalities with planning")
        # Group consecutive same prompts; within each group, generate sequentially (each sees
        # already_generated). Across groups, batch inference by round to parallelize.
        groups: list[tuple[str, list[int]]] = []  # (user_prompt, output_indices)
        prev_prompt = object()  # sentinel
        for i, user_prompt in enumerate(prompt_texts):
            if user_prompt != prev_prompt:
                groups.append((user_prompt, [i]))
            else:
                groups[-1][1].append(i)
            prev_prompt = user_prompt

        # personalities[output_idx] = generated personality
        personalities = [None] * len(prompt_texts)
        # group_personalities[g] = list of personalities generated for group g so far
        group_personalities: list[list[str]] = [[] for _ in groups]

        round_idx = 0
        while True:
            # Build batch: for each group that needs slot round_idx, build the prompt
            batch_prompts: list[str] = []
            batch_group_slot: list[tuple[int, int]] = []  # (group_idx, slot_in_group)
            for g, (user_prompt, out_indices) in enumerate(groups):
                if round_idx >= len(out_indices):
                    continue
                already = group_personalities[g]
                if len(already) > 0:
                    system_content = PERSONALITY_GEN_SYSTEM + " " + PERSONALITY_GEN_PLANNING_ADDON
                    already_str = "\n".join(f"- {p}" for p in already)
                    user_content = (
                        f"User prompt:\n\n{user_prompt or '(No specific prompt)'}\n\n"
                        f"Already generated personalities (generate something different):\n{already_str}"
                    )
                else:
                    system_content = PERSONALITY_GEN_SYSTEM
                    user_content = (
                        f"User prompt:\n\n{user_prompt}"
                        if user_prompt and user_prompt.strip()
                        else "User prompt:\n\n(No specific prompt)"
                    )
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
                prompt_text = apply_template(messages, add_generation_prompt=True, tokenize=False)
                batch_prompts.append(prompt_text)
                batch_group_slot.append((g, round_idx))

            if not batch_prompts:
                break

            generated = _run_personality_generation(
                batch_prompts, rollout_wg, tokenizer, pad_token_id, response_length
            )
            for (g, slot), personality in zip(batch_group_slot, generated, strict=True):
                group_personalities[g].append(personality)
                out_idx = groups[g][1][slot]
                personalities[out_idx] = personality

            round_idx += 1

        personalities = [p + "\n" + PERSONALITY_FINAL_OUTPUT_CONSTRAINT for p in personalities]
        print(f"[Personality] Generated {len(personalities)} personalities (batched inference)")
        return personalities

    print(f"[Personality] Generating personalities without planning")
    # Batch mode: for each unique prompt, ask model to generate N personalities in one response,
    # then parse and assign to the correct indices.
    prompt_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, user_prompt in enumerate(prompt_texts):
        key = user_prompt if (user_prompt and user_prompt.strip()) else "(No specific prompt)"
        prompt_to_indices[key].append(i)

    personality_prompts: list[str] = []
    out_indices_per_batch: list[list[int]] = []
    for user_prompt, out_indices in prompt_to_indices.items():
        n = len(out_indices)
        system_content = PERSONALITY_GEN_SYSTEM_MULTI
        user_content = (
            f"User prompt:\n\n{user_prompt}\n\n"
            f"Generate exactly {n} different personality descriptions."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        prompt_text = apply_template(messages, add_generation_prompt=True, tokenize=False)
        personality_prompts.append(prompt_text)
        out_indices_per_batch.append(out_indices)

    # Scale response_length for multi-personality (each needs ~50-80 tokens)
    max_n = max(len(o) for o in out_indices_per_batch)
    multi_response_length = max(response_length * 2, response_length * max_n)

    outputs = _run_personality_generation(
        personality_prompts, rollout_wg, tokenizer, pad_token_id, multi_response_length
    )

    # Parse each response and assign to output indices
    personalities = [None] * len(prompt_texts)
    for (out_indices, raw_text) in zip(out_indices_per_batch, outputs, strict=True):
        parsed = _parse_multi_personality_response(raw_text, len(out_indices))
        for idx, p in zip(out_indices, parsed, strict=True):
            personalities[idx] = p

    personalities = [p + "\n" + PERSONALITY_FINAL_OUTPUT_CONSTRAINT for p in personalities]
    print(f"[Personality] Generated {len(personalities)} personalities (multi-per-response mode)")
    return personalities


def _prepend_personality_to_user_content(content: Any, personality: str) -> Any:
    """Prepend personality text to the first user message body (string or multimodal list)."""
    sep = "\n\n"
    p = personality.rstrip()
    if isinstance(content, str):
        c = content.lstrip()
        return f"{p}{sep}{c}" if c else p
    if isinstance(content, list):
        out: list[Any] = []
        prepended = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                pc = dict(part)
                t = str(pc["text"]).lstrip()
                pc["text"] = f"{p}{sep}{t}" if t else p
                out.append(pc)
                prepended = True
            else:
                out.append(part)
        if not prepended:
            out.insert(0, {"type": "text", "text": p})
        return out
    c = str(content).lstrip()
    return f"{p}{sep}{c}" if c else p


def _apply_personality_to_messages(messages: list[dict], personality: str, inject_mode: str) -> None:
    """Mutate ``messages`` in place: system turn vs start of user content."""
    mode = (inject_mode or "system").strip().lower()
    if mode == "system":
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": personality}
        else:
            messages.insert(0, {"role": "system", "content": personality})
        return
    if mode == "user":
        user_i = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
        if user_i is None:
            print(
                "[Personality] inject_mode=user but no user message in chat; falling back to system injection."
            )
            _apply_personality_to_messages(messages, personality, "system")
            return
        msg = dict(messages[user_i])
        msg["content"] = _prepend_personality_to_user_content(msg.get("content", ""), personality)
        messages[user_i] = msg
        return
    raise ValueError(f"Invalid inject_mode {inject_mode!r}; expected 'system' or 'user'.")


def generate_and_inject_novel_personalities_to_batch(
    batch: DataProto,
    rollout_wg,
    tokenizer,
    processor=None,
    personality_gen_response_length: int = 128,
    planning: bool = False,
    inject_mode: str = "system",
    apply_chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> DataProto:
    """Generate novel personalities per prompt and inject them into the batch.

    1. Extracts user prompt text for each position in the batch.
    2. Prompts the model to generate a personality for each.
    3. Injects the generated personalities via inject_personality_to_batch.

    Args:
        batch: DataProto with input_ids, raw_prompt/extra_info/data_source
        rollout_wg: Worker group with generate_sequences (e.g. actor_rollout_wg)
        tokenizer: Tokenizer for encoding/decoding
        processor: Optional processor for multimodal
        personality_gen_response_length: Max tokens for personality generation
        planning: If True, generate one at a time and include already_generated_personalities
            in each prompt so the LLM produces distinctly different personalities.

        inject_mode: ``"system"`` or ``"user"`` (passed to ``inject_personality_to_batch``).
        apply_chat_template_kwargs: Extra kwargs forwarded to ``apply_chat_template`` when
            rebuilding prompts during personality injection.

    Returns:
        Modified batch with personalities injected per ``inject_mode``
    """
    batch_size = len(batch.batch["input_ids"])
    prompt_texts_with_idx = _extract_prompt_texts_from_batch(batch)

    # Build list of (text, idx) for generation; use None for fallback
    texts_to_generate = []
    indices_to_generate = []
    fallback_indices = []

    for text, idx in prompt_texts_with_idx:
        if text is not None:
            texts_to_generate.append(text)
            indices_to_generate.append(idx)
        else:
            fallback_indices.append(idx)

    # Generate personalities for valid prompts
    if texts_to_generate:
        generated = _generate_personalities_with_model(
            texts_to_generate,
            rollout_wg,
            tokenizer,
            processor,
            response_length=personality_gen_response_length,
            planning=planning,
        )
    else:
        generated = []

    # Map generated personalities back to batch indices
    idx_to_personality = dict(zip(indices_to_generate, generated, strict=True))
    default_personality = "You are a helpful and knowledgeable assistant."
    personalities = []
    for idx in range(batch_size):
        if idx in idx_to_personality:
            personalities.append(idx_to_personality[idx])
        else:
            personalities.append(default_personality)

    # Reuse inject_personality_to_batch
    return inject_personality_to_batch(
        batch,
        personalities,
        tokenizer,
        processor,
        inject_mode=inject_mode,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

def inject_personality_to_batch(
    batch: DataProto,
    personalities: list[str],
    tokenizer,
    processor=None,
    inject_mode="system",
    apply_chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> DataProto:
    """Inject personality prompts into batch data before generation.

    With ``inject_mode="system"``, the personality is set as (or merged into) the system message.
    With ``inject_mode="user"``, the personality is prepended to the first user message content.

    It assumes that the batch has already been repeated by rollout.n times, so we assign
    one personality to each repetition.

    Args:
        batch: DataProto containing the batch data with 'data_source' in non_tensor_batch
        personalities: List of personality prompts (length should match rollout.n)
        tokenizer: Tokenizer for encoding
        processor: Optional processor for multimodal data
        inject_mode: ``"system"`` or ``"user"``
        apply_chat_template_kwargs: Extra kwargs forwarded to ``apply_chat_template`` when
            rebuilding prompts.

    Returns:
        Modified DataProto with personality prompts injected
    """
    batch_size = len(batch.batch["input_ids"])
    n_personalities = len(personalities)
    print(f"[Personality] Injecting {n_personalities} personalities (inject_mode={inject_mode!r}) into batch of size {batch_size}")
    template_kwargs = apply_chat_template_kwargs or {}
    
    
    # Track which personality each sample gets
    personality_ids = [idx % n_personalities for idx in range(batch_size)]
    
    # Log available keys for debugging
    print(f"[Personality] Available batch.non_tensor_batch keys: {list(batch.non_tensor_batch.keys())}, "
          f"Available batch.batch keys: {list(batch.batch.keys())}")
    
    # Try to find messages from different sources:
    # 1. raw_prompt (if return_raw_chat=True in config)
    # 2. extra_info["question"] (for GSM8K and similar datasets)
    # 3. data_source if it contains messages (some datasets)
    
    raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
    extra_infos = batch.non_tensor_batch.get("extra_info", None)
    data_sources = batch.non_tensor_batch.get("data_source", None)
    
    # Store results indexed by original position to maintain order
    result_map = {}
    
    # Collect prompt texts for batch tokenization
    prompt_texts = []
    prompt_texts_indices = []
    fallback_indices = []
    success_count = 0

    for idx in range(batch_size):
        personality_idx = personality_ids[idx]
        messages = None
        
        # Method 1: Use raw_prompt if available (best option - contains original messages)
        if raw_prompts is not None and idx < len(raw_prompts):
            raw_prompt = raw_prompts[idx]
            if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
                messages = [dict(msg) for msg in raw_prompt]  # Deep copy
                if idx == 0:
                    print(f"[Personality] Using raw_prompt for messages")
        
        # Method 2: Reconstruct from extra_info["question"] (for GSM8K, OpenScholar)
        if messages is None and extra_infos is not None and idx < len(extra_infos):
            extra_info = extra_infos[idx]
            if isinstance(extra_info, dict):
                question = extra_info.get("question", None)
                if question is not None:
                    # Reconstruct message with the GSM8K instruction
                    instruction_following = ""
                    content = f"{question} {instruction_following}"
                    messages = [{"role": "user", "content": content}]
                    if idx == 0:
                        print(f"[Personality] Reconstructed messages from extra_info['question']")
        
        # Method 3: Check if data_source is a dict with messages (legacy)
        if messages is None and data_sources is not None and idx < len(data_sources):
            data_item = data_sources[idx]
            if isinstance(data_item, dict):
                for key in ["messages", "prompt"]:
                    if key in data_item:
                        raw_msgs = data_item[key]
                        if isinstance(raw_msgs, list) and len(raw_msgs) > 0:
                            messages = [dict(msg) for msg in raw_msgs]
                            if idx == 0:
                                print(f"[Personality] Using data_source['{key}'] for messages")
                            break
        
        if messages is not None:
            try:
                _apply_personality_to_messages(messages, personalities[personality_idx], inject_mode)

                # Build the prompt text (don't tokenize here)
                if processor is not None:
                    prompt_text = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        **template_kwargs,
                    )
                else:
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        **template_kwargs,
                    )

                prompt_texts.append(prompt_text)
                prompt_texts_indices.append(idx)
                success_count += 1
            except Exception as e:
                print(f"[Personality] Warning: Failed to build prompt for sample {idx}: {e}")
                fallback_indices.append(idx)
        else:
            # Fallback: keep original input_ids if we can't get messages
            if idx < 3:  # Log first few fallbacks
                print(f"[Personality] Sample {idx}: Could not find messages. "
                      f"raw_prompt={raw_prompts[idx] if raw_prompts is not None and idx < len(raw_prompts) else 'N/A'}, "
                      f"extra_info keys={list(extra_infos[idx].keys()) if extra_infos is not None and idx < len(extra_infos) and isinstance(extra_infos[idx], dict) else 'N/A'}")
            fallback_indices.append(idx)

    # Batch tokenize all successful prompt texts
    if prompt_texts:
        print(f"[Personality] Tokenizing {len(prompt_texts)} prompts...")
        try:
            model_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
            batch_input_ids = model_inputs["input_ids"]
            batch_attention_mask = model_inputs["attention_mask"]
            tokenized_max_len = batch_input_ids.shape[1]
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

            for i, global_idx in enumerate(prompt_texts_indices):
                row_ids = batch_input_ids[i]
                row_mask = batch_attention_mask[i]
                actual_len = int(row_mask.sum().item())
                
                if actual_len == 0:
                    # Empty, use original
                    result_map[global_idx] = None
                    continue

                # Convert right-padded to left-padded
                tokens_nonpad = row_ids[:actual_len]
                mask_nonpad = row_mask[:actual_len]
                
                # Store unpadded for now; we'll pad all to same length later
                result_map[global_idx] = (tokens_nonpad, mask_nonpad)

        except Exception as e:
            print(f"[Personality] Warning: Batch tokenization failed: {e}. Using originals for all.")
            for global_idx in prompt_texts_indices:
                result_map[global_idx] = None  # Fallback to original

    # Mark fallback indices
    for idx in fallback_indices:
        result_map[idx] = None  # Fallback to original

    # Build final tensors IN ORIGINAL ORDER
    final_input_ids_list = []
    final_attention_mask_list = []
    
    for idx in range(batch_size):
        if idx in result_map and result_map[idx] is not None:
            input_ids, attention_mask = result_map[idx]
            final_input_ids_list.append(input_ids)
            final_attention_mask_list.append(attention_mask)
        else:
            # Use original
            final_input_ids_list.append(batch.batch["input_ids"][idx])
            final_attention_mask_list.append(batch.batch["attention_mask"][idx])

    if success_count > 0:
        # Pad all sequences to same length (left-padding)
        max_len = max(len(ids) for ids in final_input_ids_list)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        padded_input_ids = []
        padded_attention_mask = []
        padded_position_ids = []

        for input_ids, attention_mask in zip(final_input_ids_list, final_attention_mask_list):
            current_len = len(input_ids)
            pad_len = max_len - current_len
            
            # Left padding (standard for generation)
            padded_input_ids.append(
                torch.cat([torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype), input_ids])
            )
            padded_attention_mask.append(
                torch.cat([torch.zeros(pad_len, dtype=attention_mask.dtype), attention_mask])
            )
            # Position IDs: zero for padding, then sequential
            position_ids = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.arange(current_len, dtype=torch.long)
            ])
            padded_position_ids.append(position_ids)

        # Update batch with modified inputs
        batch.batch["input_ids"] = torch.stack(padded_input_ids)
        batch.batch["attention_mask"] = torch.stack(padded_attention_mask)
        batch.batch["position_ids"] = torch.stack(padded_position_ids)
        
        print(f"[Personality] Successfully injected personalities into {success_count}/{batch_size} samples")
        
        # Log a sample prompt to verify injection
        sample_prompt = tokenizer.decode(batch.batch["input_ids"][0], skip_special_tokens=True)
        print(f"[Personality] Sample prompt (first 300 chars): {sample_prompt[:300]}...")
    else:
        print(f"[Personality] Warning: No samples were successfully processed!")

    # Store personality IDs for later use in diversity computation
    batch.non_tensor_batch["personality_id"] = np.array(personality_ids, dtype=np.int32)
    batch.non_tensor_batch['raw_prompt_ids'] = np.array([t.numpy().astype(np.int32) for t in final_input_ids_list], dtype=object)
    print(f"[Personality] Assigned personality IDs: unique={np.unique(personality_ids)}")
    return batch


def compute_diversity_reward(
    data: DataProto,
    tokenizer,
    metric: str = "token_overlap",
    diversity_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute diversity-based reward bonus for responses from different personalities.

    For each original prompt (identified by uid), we compute pairwise diversity
    between the responses from different personalities. Higher diversity gets higher reward.

    NOTE: The main compute_diversity_reward used by ray_trainer.py is in 
    custom_reward_setup/custom_reward_function.py. This is a fallback/alternative version.

    Args:
        data: DataProto containing responses and uid grouping
        tokenizer: Tokenizer for decoding responses
        metric: Diversity metric to use ("token_overlap" or "edit_distance")
        diversity_weight: Weight for diversity reward

    Returns:
        Tuple of (diversity_rewards tensor matching response shape, metrics dict)
    """
    responses = data.batch["responses"]  # (batch_size, response_length)
    batch_size = responses.shape[0]
    response_length = responses.shape[1]
    
    if "uid" not in data.non_tensor_batch:
        print("[Diversity] Warning: 'uid' not found, cannot compute diversity reward")
        return torch.zeros(batch_size, response_length, device=responses.device, dtype=torch.float32), {}

    uids = data.non_tensor_batch["uid"]

    # Group responses by uid
    uid_to_indices = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[uid].append(idx)

    # Initialize per-sample diversity rewards
    diversity_rewards_per_sample = torch.zeros(batch_size, device=responses.device)
    diversity_scores = []

    # For each group (original prompt), compute pairwise diversity
    for uid, indices in uid_to_indices.items():
        if len(indices) <= 1:
            continue

        # Decode responses for this group
        group_responses = [responses[idx] for idx in indices]
        decoded_responses = tokenizer.batch_decode(group_responses, skip_special_tokens=True)

        # Compute pairwise diversity
        n_responses = len(decoded_responses)
        pairwise_diversities = []

        for i in range(n_responses):
            for j in range(i + 1, n_responses):
                diversity = compute_pairwise_diversity(
                    decoded_responses[i],
                    decoded_responses[j],
                    metric=metric
                )
                pairwise_diversities.append(diversity)

        # Average diversity for this group
        if pairwise_diversities:
            avg_diversity = np.mean(pairwise_diversities)
            diversity_scores.append(avg_diversity)

            # Assign the same diversity reward to all responses in this group
            for idx in indices:
                diversity_rewards_per_sample[idx] = avg_diversity * diversity_weight

    # Create token-level diversity rewards matching the response tensor shape
    # Place diversity reward at the last valid token position (matching accuracy reward placement)
    diversity_rewards = torch.zeros(batch_size, response_length, device=responses.device, dtype=torch.float32)
    
    if "response_mask" in data.batch:
        response_mask = data.batch["response_mask"]
        valid_lengths = response_mask.sum(dim=-1).long() - 1
        for idx in range(batch_size):
            if 0 <= valid_lengths[idx] < response_length:
                diversity_rewards[idx, valid_lengths[idx]] = diversity_rewards_per_sample[idx]
    else:
        # Fallback: place at last position
        diversity_rewards[:, -1] = diversity_rewards_per_sample

    metrics = {
        "diversity/mean_score": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        "diversity/num_groups": len(uid_to_indices),
    }

    return diversity_rewards, metrics


def compute_pairwise_diversity(text1: str, text2: str, metric: str = "token_overlap") -> float:
    """Compute diversity between two text strings.

    Args:
        text1: First text string
        text2: Second text string
        metric: Metric to use ("token_overlap" or "edit_distance")

    Returns:
        Diversity score (higher = more diverse)
    """
    if metric == "token_overlap":
        # Use 1 - Jaccard similarity as diversity
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())

        if not tokens1 and not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        if union == 0:
            return 0.0

        jaccard_similarity = intersection / union
        diversity = 1.0 - jaccard_similarity

        return diversity

    elif metric == "edit_distance":
        # Normalized edit distance as diversity
        import Levenshtein

        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 0.0

        edit_dist = Levenshtein.distance(text1, text2)
        normalized_diversity = edit_dist / max_len

        return normalized_diversity

    else:
        raise ValueError(f"Unknown diversity metric: {metric}")

