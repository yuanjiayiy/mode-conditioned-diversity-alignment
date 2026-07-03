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
Compute base-model Skyreward reference scores for Infinite Chats Taxonomy.

For each prompt, samples n_samples responses from the base model, evaluates them
with Skyreward, and saves the average as reward_model["reference_score"].

Usage:
    python infinite_chats_taxonomy_reference_scores.py \\
        --input_dir ~/data/infinite-chats-taxonomy \\
        --base_model_path Qwen/Qwen3-8B-Base \\
        [--n_samples 5] [--gen_batch_size 8] [--output_dir ...]
"""

import argparse
import contextlib
import gc
import json
import numpy as np
import os
import sys

from verl.utils.nlp_utils import _text_after_thinking

# Add project root for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, "..", "..")
sys.path.insert(0, os.path.abspath(_project_root))

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
import torch

from custom_reward_setup.infinite_chat_reward_function import (
    initialize_skyreward_model,
    compute_skyreward_reward_batch,
)


def release_vllm_gpu(llm):
    """
    Release vLLM GPU memory so another model (e.g. Skyreward) can be loaded.
    Based on: https://github.com/vllm-project/vllm/issues/6544
    """
    del llm
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    print("Released vLLM GPU memory.")


def get_gpu_count():
    return torch.cuda.device_count()

def extract_question_from_prompt(prompt_row):
    """Extract question string from prompt (list of chat dicts)."""
    if isinstance(prompt_row, (list, np.ndarray)) and len(prompt_row) > 0:
        first = prompt_row[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"]).strip()
    return ""


def _split_has_reference_responses(df, n_samples, column_name="extra_info"):
    """Check if all rows have extra_info['reference_generated_responses'] with n_samples items."""
    if column_name not in df.columns:
        return False
    for idx in range(len(df)):
        extra = df.at[idx, column_name]
        if not isinstance(extra, dict):
            extra = dict(extra) if hasattr(extra, "items") else {}
        ref_resp = extra.get("reference_generated_responses", [])
        if len(ref_resp) != n_samples:
            return False
    return True


def _save_and_push_split(df, output_dir, split, push_to_hub, dataset_name):
    """Save split parquet and optionally push to HuggingFace."""
    out_path = os.path.join(output_dir, f"{split}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}")
    if push_to_hub and dataset_name:
        try:
            dataset = Dataset.from_pandas(df)
            dataset.push_to_hub(dataset_name, split=split)
            print(f"  Pushed to {dataset_name} ({split})")
        except Exception as e:
            print(f"  Error pushing to {dataset_name} ({split}): {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute base-model Skyreward reference scores for Infinite Chats Taxonomy"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing train.parquet and validation.parquet",
    )
    parser.add_argument(
        "--base_model_path",
        required=True,
        help="HuggingFace model path for base model (e.g. Qwen/Qwen3-8B-Base)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: overwrite input_dir)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of base-model samples per prompt",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=32,
        help="Batch size for vLLM generation",
    )
    parser.add_argument(
        "--skyreward_batch_size",
        type=int,
        default=32,
        help="Batch size for Skyreward scoring",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens per generated response",
    )
    parser.add_argument(
        "--max_tokens_for_thinking",
        type=int,
        default=0,
        help="Additional tokens reserved for a reasoning trace (0 disables reasoning trace)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for base model",
    )
    parser.add_argument(
        "--skyreward_model",
        default="Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
        help="Skyreward model name",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "train"],
        help="Parquet splits to process",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push to Hub",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Dataset name",
    )
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else input_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load base model tokenizer for chat formatting
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 1: Generate for splits that don't have reference_generated_responses
    split_outputs = {}  # split -> (responses_flat, prompts_flat, n_prompts, df)
    splits_needing_generation = []
    
    # Update extra_info["reference_score"]
    column_name = "extra_info_testing"
    # Store both reference_generated_responses and reference_score in extra_info when reasoning trace is enabled

    for split in args.splits:
        parquet_path = os.path.join(input_dir, f"{split}.parquet")
        if not os.path.exists(parquet_path):
            print(f"Skipping {split}: {parquet_path} not found")
            continue

        df = pd.read_parquet(parquet_path)
        prompt_key = "prompt" if "prompt" in df.columns else "prompts"
        if prompt_key not in df.columns:
            raise ValueError(f"No prompt column found. Columns: {list(df.columns)}")

        if _split_has_reference_responses(df, args.n_samples, column_name=column_name):
            print(f"\n[Phase 1] Skipping {split}: reference_generated_responses already exists")
            questions = [extract_question_from_prompt(row[prompt_key]) for _, row in df.iterrows()]
            responses_flat = []
            prompts_flat = []
            for idx in range(len(df)):
                extra = df.at[idx, column_name]
                if not isinstance(extra, dict):
                    extra = dict(extra) if hasattr(extra, "items") else {}
                ref_resp = extra.get("reference_generated_responses", [])
                responses_flat.extend(ref_resp)
                prompts_flat.extend([questions[idx]] * len(ref_resp))
            split_outputs[split] = (responses_flat, prompts_flat, len(questions), df)
        else:
            splits_needing_generation.append((split, df))

    if splits_needing_generation:
        # Load vLLM and generate for splits that need it
        print(f"\nLoading base model: {args.base_model_path}")
        llm = LLM(
            model=args.base_model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=get_gpu_count(),
        )
        # If max_tokens_for_thinking > 0 we enable the generation prompt (reasoning trace)
        # and reserve extra tokens for it. When set to 0, the reasoning trace is disabled.
        gen_max_tokens = args.max_tokens + max(0, args.max_tokens_for_thinking)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=gen_max_tokens,
            n=args.n_samples,
        )

        for split, df in splits_needing_generation:
            prompt_key = "prompt" if "prompt" in df.columns else "prompts"
            questions = [extract_question_from_prompt(row[prompt_key]) for _, row in df.iterrows()]
            n_prompts = len(questions)
            print(f"\n[Phase 1] Generating for {split}... ({n_prompts} prompts)")

            formatted_prompts = []
            enable_thinking = args.max_tokens_for_thinking > 0
            for q in questions:
                messages = [{"role": "user", "content": q}]
                prompt_str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                formatted_prompts.append(prompt_str)

            all_outputs = []
            for i in tqdm(range(0, n_prompts, args.gen_batch_size), desc="Generating"):
                batch_prompts = formatted_prompts[i : i + args.gen_batch_size]
                outputs = llm.generate(batch_prompts, sampling_params)
                for out in outputs:
                    texts = [o.text for o in out.outputs]
                    all_outputs.extend(texts)

            responses_flat = all_outputs
            prompts_flat = []
            for i in range(n_prompts):
                prompts_flat.extend([questions[i]] * args.n_samples)

            split_outputs[split] = (responses_flat, prompts_flat, n_prompts, df)

            # Print first 5 generated rows immediately after generation (before saving)
            try:
                n_show = min(5, n_prompts)
                print(f"  Preview first {n_show} generated rows for {split} (before saving):")
                for ridx in range(n_show):
                    prompt_text = questions[ridx]
                    start = ridx * args.n_samples
                    end = start + args.n_samples
                    samples = responses_flat[start:end]
                    print(f"    [{ridx}] Prompt: {prompt_text}")
                    for sidx, sample in enumerate(samples):
                        out_preview = sample.replace("\n", " ")[:500]
                        print(f"      - Sample {sidx}: {out_preview}")
                    print("")
            except Exception as e:
                print(f"  Warning: failed to print preview rows: {e}")

            # After Phase 1: save reference_generated_responses to output and upload to HuggingFace
            if column_name not in df.columns:
                df[column_name] = [{} for _ in range(len(df))]
            for i in range(n_prompts):
                start = i * args.n_samples
                end = start + args.n_samples
                extra_info = df.at[i, column_name]
                if isinstance(extra_info, dict):
                    extra_info = extra_info.copy()
                else:
                    extra_info = dict(extra_info) if hasattr(extra_info, "items") else {}
                extra_info["reference_generated_responses"] = responses_flat[start:end]
                df.at[i, column_name] = extra_info

            print(f"  Saving {split} with reference_generated_responses...")
            _save_and_push_split(df, output_dir, split, args.push_to_hub, args.dataset_name)

        # Release vLLM GPU memory before loading Skyreward
        print("\nReleasing vLLM GPU memory...")
        release_vllm_gpu(llm)

    # Phase 2: Load Skyreward and score all splits
    print(f"\nLoading Skyreward model: {args.skyreward_model}")
    initialize_skyreward_model(args.skyreward_model)
    
    responses_flat = [_text_after_thinking(r) for r in responses_flat]

    for split, (responses_flat, prompts_flat, n_prompts, df) in split_outputs.items():
        print(f"\n[Phase 2] Scoring {split}...")
        n_total = len(responses_flat)
        scores_flat = []
        for j in tqdm(range(0, n_total, args.skyreward_batch_size), desc="Scoring"):
            batch_resp = responses_flat[j : j + args.skyreward_batch_size]
            batch_prom = prompts_flat[j : j + args.skyreward_batch_size]
            batch_scores = compute_skyreward_reward_batch(batch_resp, batch_prom)
            scores_flat.extend(batch_scores)

        # Average per prompt
        reference_scores = []
        reference_min_scores = []
        reference_max_scores = []
        reference_generated_responses = []
        for i in range(n_prompts):
            start = i * args.n_samples
            end = start + args.n_samples
            avg = sum(scores_flat[start:end]) / args.n_samples
            min_score = min(scores_flat[start:end])
            max_score = max(scores_flat[start:end])
            reference_scores.append(avg)
            reference_min_scores.append(min_score)
            reference_max_scores.append(max_score)
            reference_generated_responses.append(responses_flat[start:end])

        
        if column_name not in df.columns:
            df[column_name] = [{} for _ in range(len(df))]
        for idx, ref_score in enumerate(reference_scores):
            extra_info = df.at[idx, column_name]
            if isinstance(extra_info, dict):
                extra_info = extra_info.copy()
            else:
                extra_info = dict(extra_info) if hasattr(extra_info, "items") else {}
            extra_info["reference_mean_score"] = ref_score
            extra_info["reference_min_score"] = reference_min_scores[idx]
            extra_info["reference_max_score"] = reference_max_scores[idx]
            extra_info["reference_generated_responses"] = reference_generated_responses[idx]
            df.at[idx, column_name] = extra_info

        _save_and_push_split(df, output_dir, split, args.push_to_hub, args.dataset_name)

    print("\nDone.")

if __name__ == "__main__":
    main()
