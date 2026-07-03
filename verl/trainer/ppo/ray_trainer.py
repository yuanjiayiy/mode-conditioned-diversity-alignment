# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

from itertools import islice
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import RewardFnActor, compute_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.nlp_utils import _conceptual_density_proxy, _emotion_category_rates, _load_category_lexicon, _readability_metrics, _simple_word_tokenize, _vocab_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.personality_diversity import (
    generate_default_personalities,
    inject_personality_to_batch,
    # compute_diversity_reward,
)

from custom_reward_setup.custom_reward_function import compute_diversity_reward

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        
        # Log reward function information for verification
        print("\n" + "="*80)
        print("REWARD FUNCTION SETUP:")
        print("="*80)
        if self.reward_fn is not None:
            reward_fn_class = type(self.reward_fn).__name__
            print(f"Training reward_fn: {reward_fn_class}")
            if hasattr(self.reward_fn, 'compute_score'):
                compute_score_fn = self.reward_fn.compute_score
                compute_score_name = getattr(compute_score_fn, '__name__', 'unknown')
                compute_score_module = getattr(compute_score_fn, '__module__', 'unknown')
                compute_score_str = str(compute_score_fn)
                print(f"  compute_score function: {compute_score_name} (from {compute_score_module})")
                
                # Check if it's math reward - check multiple indicators
                math_indicators = [
                    'math' in compute_score_name.lower(),
                    'compute_math_reward' in compute_score_str,
                    'hybrid_reward_function' in compute_score_str,
                    'compute_math_reward' in compute_score_module,
                ]
                
                if any(math_indicators):
                    print(f"  ✓ Math reward (regex heuristic) detected!")
                    print(f"    (Based on: function name or module containing 'math' or 'hybrid_reward_function')")
                elif 'default_compute_score' in compute_score_str:
                    print(f"  → Using default_compute_score (may use math reward based on data_source)")
                else:
                    print(f"  → Using custom reward function (check implementation for math reward)")
            print(f"  use_rm (reward model): {self.config.reward_model.get('enable', False)}")
            if self.config.reward_model.get('enable', False):
                print(f"  ⚠ WARNING: Reward model is enabled! Math reward (regex) may not be used.")
            else:
                print(f"  ✓ Reward model disabled - using rule-based reward_fn")
        else:
            print("Training reward_fn: None (will use reward model if use_rm=True)")
        
        if self.val_reward_fn is not None:
            val_reward_fn_class = type(self.val_reward_fn).__name__
            print(f"Validation reward_fn: {val_reward_fn_class}")
        else:
            print("Validation reward_fn: None")
        print("="*80 + "\n")
        
        self.reward_fn_actor = None
        if self.config.reward_model.launch_reward_fn_async:
            async_cpus = self.config.reward_model.get("async_num_cpus", 1)
            async_gpus = self.config.reward_model.get("async_num_gpus", 0)
            self.reward_fn_actor = RewardFnActor.options(
                num_cpus=async_cpus,
                num_gpus=async_gpus,
            ).remote(self.config, tokenizer)

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # Initialize personalities for diversity-based generation
        self.use_personality_diversity = self.config.algorithm.get("use_personality_diversity", False)
        if self.use_personality_diversity:
            n_rollout = self.config.actor_rollout_ref.rollout.n
            personalities = self.config.algorithm.get("personalities", None)
            
            # Generate or validate personalities
            if personalities is None or len(personalities) != n_rollout:
                if personalities is not None:
                    print(f"Warning: personalities list length ({len(personalities)}) doesn't match rollout.n ({n_rollout})")
                print(f"Generating {n_rollout} default personalities")
                personalities = generate_default_personalities(n_rollout, domain=self.config.algorithm.get("personality_domain", "general"))
            
            self.personalities = personalities
            print(f"Using personality-based generation with {len(self.personalities)} personalities:")
            for i, p in enumerate(self.personalities):
                print(f"  Personality {i+1}: {p}")
        else:
            self.personalities = None

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            # "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_responses_readable(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, prefix="rollout", personality_ids=None):
        """Dump responses in a human-readable format for easy inspection."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{prefix}_responses_step_{self.global_steps}.txt")

        n = len(inputs)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=" * 100 + "\n")
            f.write(f"Step {self.global_steps} - {prefix.upper()} Responses\n")
            f.write(f"=" * 100 + "\n\n")
            
            # Summary of personalities if available
            if personality_ids is not None:
                unique_pids = np.unique(personality_ids)
                f.write(f"Personality IDs in batch: {unique_pids}\n")
                for pid in unique_pids:
                    count = sum(1 for p in personality_ids if p == pid)
                    f.write(f"  Personality {pid}: {count} samples\n")
                f.write("\n")
            
            for i in range(n):
                f.write(f"\n--- Sample {i+1}/{n} ---\n")
                f.write(f"Step: {self.global_steps}\n")
                
                # Show personality ID if available
                if personality_ids is not None and i < len(personality_ids):
                    f.write(f"Personality ID: {personality_ids[i]}\n")
                    if self.personalities and personality_ids[i] < len(self.personalities):
                        f.write(f"Personality: {self.personalities[personality_ids[i]][:100]}...\n")
                
                f.write(f"Score: {scores[i]:.4f}\n")
                
                # Add reward breakdown if available
                if reward_extra_infos_dict:
                    f.write("Reward Breakdown:\n")
                    for key, val_list in reward_extra_infos_dict.items():
                        if len(val_list) > i:
                            if isinstance(val_list[i], (int, float)):
                                f.write(f"  {key}: {val_list[i]:.4f}\n")
                            else:
                                f.write(f"  {key}: {val_list[i]}\n")
                
                f.write(f"\nInput (check for personality in system prompt):\n{inputs[i]}\n\n")
                # f.write(f"Ground Truth: {gts[i] if gts[i] is not None else 'N/A'}\n\n")
                f.write(f"Generated Output:\n{outputs[i]}\n")
                f.write("\n" + "-" * 100 + "\n")
            
            f.write(f"\nTotal samples: {n}\n")
            if scores:
                f.write(f"Average score: {sum(scores)/len(scores):.4f}\n")
                f.write(f"Max score: {max(scores):.4f}\n")
                f.write(f"Min score: {min(scores):.4f}\n")

        print(f"Dumped readable responses to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            # Get personality_ids if available
            personality_ids = batch.non_tensor_batch.get("personality_id", None)
            
            # Dump JSONL format (for programmatic access)
            # Include personality_id in the dump
            if personality_ids is not None:
                reward_extra_infos_to_dump["personality_id"] = personality_ids.tolist() if hasattr(personality_ids, 'tolist') else list(personality_ids)
            
            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
            
            # Dump human-readable format (for easy inspection)
            readable_dir = os.path.join(rollout_data_dir, "readable")
            self._dump_responses_readable(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=readable_dir,
                prefix="rollout",
                personality_ids=personality_ids,
            )

            # Optional: upload rollout data dir to wandb as an artifact
            upload_rollout = self.config.trainer.get("wandb_upload_rollout_data", False)
            rollout_upload_freq = int(self.config.trainer.get("wandb_rollout_upload_freq", 0) or 0)
            if upload_rollout and hasattr(self, "tracking") and self.tracking is not None:
                if rollout_upload_freq <= 0 or (self.global_steps % rollout_upload_freq == 0):
                    art_name = self.config.trainer.get(
                        "wandb_rollout_artifact_name",
                        f"{self.config.trainer.experiment_name}-rollout",
                    )
                    self.tracking.log_artifact(
                        path=rollout_data_dir,
                        name=art_name,
                        artifact_type=self.config.trainer.get("wandb_rollout_artifact_type", "dataset"),
                        aliases=["latest", f"step-{self.global_steps}"],
                        metadata={"global_step": int(self.global_steps)},
                    )

    def _compute_personality_metrics(
        self, 
        batch: DataProto, 
        reward_tensor: torch.Tensor,
        diversity_metrics: dict,
    ) -> dict:
        """
        Compute per-personality statistics for analysis.
        
        Returns a dict containing:
        - per_personality_scores: {personality_id: mean_score}
        - per_personality_counts: {personality_id: count}
        - separation_metrics: metrics measuring personality distinctiveness
        """
        metrics = {}
        
        if "personality_id" not in batch.non_tensor_batch:
            return metrics
        
        personality_ids = batch.non_tensor_batch["personality_id"]

        # --- NEW: decode responses once (for linguistic metrics) ---
        responses = batch.batch.get("responses", None)
        if responses is None:
            return metrics

        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        # Token length from response_mask (fast + consistent with training)
        response_mask = batch.batch.get("response_mask", None)
        if response_mask is not None:
            resp_len_tokens = response_mask.sum(dim=-1).detach().cpu().numpy().astype(int)
        else:
            # fallback: approximate from non-pad tokens
            resp_len_tokens = (responses != self.tokenizer.pad_token_id).sum(dim=-1).detach().cpu().numpy().astype(int)

        # Optional emotion/LIWC-like lexicon path in config (user-provided)
        # Example config:
        # algorithm:
        #   category_lexicon_path: "/path/to/lexicon.json"
        lex_path = self.config.algorithm.get("category_lexicon_path", None)
        cat2words = _load_category_lexicon(lex_path)

        # Compute per-sample linguistic metrics
        per_sample = {
            "resp_len_tokens": [],
            "resp_len_chars": [],
            "resp_len_words": [],
            "type_token_ratio": [],
            "lexical_entropy": [],
            "avg_word_len": [],
            "flesch_reading_ease": [],
            "fk_grade": [],
            "conceptual_density": [],
        }
        # Emotion category rates get dynamic keys: emotion_rate/<cat>
        emotion_keys = set()

        per_sample_emotion: list[dict[str, float]] = []
        for txt in response_texts:
            words = _simple_word_tokenize(txt)
            v = _vocab_metrics(words)
            r = _readability_metrics(txt)
            cd = _conceptual_density_proxy(words)
            emo = _emotion_category_rates(words, cat2words)

            per_sample["resp_len_chars"].append(len(txt or ""))
            per_sample["resp_len_words"].append(v["word_count"])
            per_sample["type_token_ratio"].append(v["type_token_ratio"])
            per_sample["lexical_entropy"].append(v["lexical_entropy"])
            per_sample["avg_word_len"].append(v["avg_word_len"])
            per_sample["flesch_reading_ease"].append(r["flesch_reading_ease"])
            per_sample["fk_grade"].append(r["fk_grade"])
            per_sample["conceptual_density"].append(cd)

            per_sample_emotion.append(emo)
            emotion_keys.update(emo.keys())

        per_sample["resp_len_tokens"] = resp_len_tokens.tolist()

        # --- existing reward aggregation ---
        if reward_tensor.dim() > 1:
            seq_rewards = reward_tensor.sum(dim=-1).cpu().numpy()
        else:
            seq_rewards = reward_tensor.cpu().numpy()

        accuracy_rewards = None
        if "accuracy_reward" in batch.non_tensor_batch:
            accuracy_rewards = batch.non_tensor_batch["accuracy_reward"]

        # Group by personality
        personality_scores = defaultdict(list)
        personality_accuracy = defaultdict(list)

        # NEW: group linguistic metrics by personality
        personality_ling = defaultdict(lambda: defaultdict(list))

        for idx, pid in enumerate(personality_ids):
            pid = int(pid)
            personality_scores[pid].append(float(seq_rewards[idx]))
            if accuracy_rewards is not None:
                personality_accuracy[pid].append(float(accuracy_rewards[idx]))

            # linguistic
            for k, arr in per_sample.items():
                personality_ling[pid][k].append(float(arr[idx]))
            for ek in emotion_keys:
                personality_ling[pid][ek].append(float(per_sample_emotion[idx].get(ek, 0.0)))

        # Compute per-personality statistics
        per_personality_data = {}
        for pid in sorted(personality_scores.keys()):
            scores = personality_scores[pid]
            per_personality_data[f"personality_{pid}"] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)) if len(scores) > 1 else 0.0,
                "count": len(scores),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
            }

            # Add accuracy if available
            if pid in personality_accuracy:
                acc_scores = personality_accuracy[pid]
                per_personality_data[f"personality_{pid}"]["mean_accuracy"] = float(np.mean(acc_scores))

            # NEW: add aggregated linguistic stats (means)
            ling_means = {}
            for k, vals in personality_ling[pid].items():
                if len(vals) > 0:
                    ling_means[f"mean_{k}"] = float(np.mean(vals))
            per_personality_data[f"personality_{pid}"].update(ling_means)

            # Wandb-friendly metrics (per personality)
            metrics[f"personality/score_mean_p{pid}"] = float(np.mean(scores))
            if accuracy_rewards is not None and pid in personality_accuracy:
                metrics[f"personality/accuracy_mean_p{pid}"] = float(np.mean(personality_accuracy[pid]))

            # NEW: log a few key linguistic metrics per personality
            for k in (
                "resp_len_tokens",
                "resp_len_words",
                "type_token_ratio",
                "fk_grade",
                "flesch_reading_ease",
                "conceptual_density",
            ):
                vals = personality_ling[pid].get(k, [])
                if vals:
                    metrics[f"personality/{k}_mean_p{pid}"] = float(np.mean(vals))

            # If lexicon provided, log per-category emotion rates too (per personality)
            for ek in sorted(emotion_keys):
                vals = personality_ling[pid].get(ek, [])
                if vals:
                    metrics[f"personality/{ek}_mean_p{pid}"] = float(np.mean(vals))
        
        
        # Personality variance/separation metrics across personalities
        for k in (
                "resp_len_tokens",
                "resp_len_words",
                "type_token_ratio",
                "fk_grade",
                "flesch_reading_ease",
                "conceptual_density",
            ):
                # Compute inter-personality variance and range for each linguistic metric
                vals_by_pid = [personality_ling[pid].get(k, []) for pid in sorted(personality_ling.keys())]
                # Take mean for each personality (ignoring empty)
                means = [float(np.mean(vals)) for vals in vals_by_pid if len(vals) > 0]
                if len(means) > 1:
                    metrics[f"personality/{k}_variance_across"] = float(np.var(means))
                    metrics[f"personality/{k}_range"] = float(max(means) - min(means))
                    
                    
        # NEW: overall (batch-level) linguistic means (helps monitoring without per-personality explosion)
        metrics["ling/resp_len_tokens_mean"] = float(np.mean(per_sample["resp_len_tokens"])) if per_sample["resp_len_tokens"] else 0.0
        metrics["ling/resp_len_words_mean"] = float(np.mean(per_sample["resp_len_words"])) if per_sample["resp_len_words"] else 0.0
        metrics["ling/type_token_ratio_mean"] = float(np.mean(per_sample["type_token_ratio"])) if per_sample["type_token_ratio"] else 0.0
        metrics["ling/fk_grade_mean"] = float(np.mean(per_sample["fk_grade"])) if per_sample["fk_grade"] else 0.0
        metrics["ling/flesch_reading_ease_mean"] = float(np.mean(per_sample["flesch_reading_ease"])) if per_sample["flesch_reading_ease"] else 0.0
        metrics["ling/conceptual_density_mean"] = float(np.mean(per_sample["conceptual_density"])) if per_sample["conceptual_density"] else 0.0
        for ek in sorted(emotion_keys):
            vals = [d.get(ek, 0.0) for d in per_sample_emotion]
            metrics[f"ling/{ek}_mean"] = float(np.mean(vals)) if vals else 0.0

        # --- existing separation metrics / diversity metrics ---
        personality_means = [per_personality_data[f"personality_{pid}"]["mean_score"] 
                           for pid in sorted(personality_scores.keys())]
        if len(personality_means) > 1:
            metrics["personality/score_variance_across"] = float(np.var(personality_means))
            metrics["personality/score_range"] = float(max(personality_means) - min(personality_means))

        if diversity_metrics:
            metrics["personality/diversity_score"] = diversity_metrics.get("diversity/mean_score", 0.0)
            metrics["diversity/within_personality_similarity"] = diversity_metrics.get("diversity/within_personality_similarity", 0.0)
            metrics["diversity/between_personality_similarity"] = diversity_metrics.get("diversity/between_personality_similarity", 0.0)
            metrics["diversity/separation_score"] = diversity_metrics.get("diversity/separation_score", 0.0)

        metrics["_per_personality_data"] = per_personality_data
        return metrics

    def _save_personality_analysis(
        self, 
        personality_metrics: dict, 
        output_dir: str,
    ):
        """
        Save personality analysis to a JSON file that accumulates over training.
        
        The JSON file has structure:
        {
            "steps": [1, 100, 200, ...],
            "personality_0": {"scores": [...], "accuracy": [...], ...},
            "personality_1": {"scores": [...], "accuracy": [...], ...},
            ...
            "separation_metrics": {"diversity": [...], "score_variance": [...], ...}
        }
        """
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "personality_analysis.json")
        
        # Load existing data or create new
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                analysis_data = json.load(f)
        else:
            analysis_data = {
                "steps": [],
                "separation_metrics": {
                    "diversity_score": [],
                    "score_variance_across": [],
                    "score_range": [],
                    "within_personality_similarity": [],
                    "between_personality_similarity": [],
                    "separation_score": [],
                },
            }
        
        # Add current step
        analysis_data["steps"].append(self.global_steps)
        
        # Extract per-personality data
        per_personality_data = personality_metrics.get("_per_personality_data", {})
        
        for key, data in per_personality_data.items():
            # key is like "personality_0"
            if key not in analysis_data:
                analysis_data[key] = {
                    "mean_scores": [],
                    "std_scores": [],
                    "counts": [],
                    "mean_accuracy": [],
                }
            
            analysis_data[key]["mean_scores"].append(data.get("mean_score", 0.0))
            analysis_data[key]["std_scores"].append(data.get("std_score", 0.0))
            analysis_data[key]["counts"].append(data.get("count", 0))
            if "mean_accuracy" in data:
                analysis_data[key]["mean_accuracy"].append(data["mean_accuracy"])
        
        # Add separation metrics
        analysis_data["separation_metrics"]["diversity_score"].append(
            personality_metrics.get("personality/diversity_score", 0.0)
        )
        analysis_data["separation_metrics"]["score_variance_across"].append(
            personality_metrics.get("personality/score_variance_across", 0.0)
        )
        analysis_data["separation_metrics"]["score_range"].append(
            personality_metrics.get("personality/score_range", 0.0)
        )
        # New separation metrics from diversity computation
        analysis_data["separation_metrics"]["within_personality_similarity"].append(
            personality_metrics.get("diversity/within_personality_similarity", 0.0)
        )
        analysis_data["separation_metrics"]["between_personality_similarity"].append(
            personality_metrics.get("diversity/between_personality_similarity", 0.0)
        )
        analysis_data["separation_metrics"]["separation_score"].append(
            personality_metrics.get("diversity/separation_score", 0.0)
        )
        
        # Save
        with open(json_path, "w") as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"[Personality Analysis] Saved to {json_path}")
        
        # Also print a summary
        print(f"[Personality Analysis] Step {self.global_steps}:")
        for key, data in per_personality_data.items():
            acc_str = f", acc={data.get('mean_accuracy', 'N/A'):.3f}" if "mean_accuracy" in data else ""
            print(f"  {key}: score={data['mean_score']:.4f} ± {data['std_score']:.4f} (n={data['count']}){acc_str}")
        within_sim = personality_metrics.get('diversity/within_personality_similarity', 0)
        between_sim = personality_metrics.get('diversity/between_personality_similarity', 0)
        sep_score = personality_metrics.get('diversity/separation_score', 0)
        print(f"  Separation: diversity={personality_metrics.get('personality/diversity_score', 0):.4f}, "
              f"within_sim={within_sim:.4f}, between_sim={between_sim:.4f}, sep_score={sep_score:.4f}")
        print(f"  Score variance across personalities: {personality_metrics.get('personality/score_variance_across', 0):.6f}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []


        n_batches = self.config.trainer.eval_sample_num // self.val_dataloader.batch_size
        for test_batch in islice(self.val_dataloader, n_batches):
            if isinstance(test_batch, dict):
                test_batch = DataProto.from_single_dict(test_batch)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # add uid to batch (same order as fit)
            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            )

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            rollout_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_gen_batch = test_gen_batch.repeat(repeat_times=rollout_n, interleave=True)

            # Match training: repeat non-tensor fields then inject personalities
            personality_ids = None
            if self.use_personality_diversity and self.personalities is not None:
                if "data_source" in test_batch.non_tensor_batch:
                    original_data_source = test_batch.non_tensor_batch["data_source"]
                    repeated_data_source = np.repeat(original_data_source, rollout_n)
                    test_gen_batch.non_tensor_batch["data_source"] = repeated_data_source
                if "extra_info" in test_batch.non_tensor_batch:
                    original_extra_info = test_batch.non_tensor_batch["extra_info"]
                    repeated_extra_info = np.repeat(original_extra_info, rollout_n)
                    test_gen_batch.non_tensor_batch["extra_info"] = repeated_extra_info
                if "raw_prompt" in test_batch.non_tensor_batch:
                    original_raw_prompt = test_batch.non_tensor_batch["raw_prompt"]
                    repeated_raw_prompt = np.repeat(original_raw_prompt, rollout_n)
                    test_gen_batch.non_tensor_batch["raw_prompt"] = repeated_raw_prompt
                elif "raw_prompt" not in test_gen_batch.non_tensor_batch:
                    print(
                        "[Personality][val] Warning: raw_prompt missing on gen batch; "
                        f"test_batch keys kept after pop: {list(test_batch.non_tensor_batch.keys())}"
                    )
                test_gen_batch = inject_personality_to_batch(
                    test_gen_batch,
                    self.personalities,
                    self.tokenizer,
                    self.processor,
                    inject_mode=self.config.algorithm.get("personality_inject_mode", "system"),
                    apply_chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}),
                )
                if "personality_id" in test_gen_batch.non_tensor_batch:
                    personality_ids = test_gen_batch.non_tensor_batch["personality_id"].copy()

            # Log inputs that match what generation actually sees (post personality injection)
            input_ids = test_gen_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # Align full batch with repeated rollouts (same as fit)
            test_batch = test_batch.repeat(repeat_times=rollout_n, interleave=True)
            if personality_ids is not None:
                test_batch.non_tensor_batch["personality_id"] = personality_ids

            sample_gts.extend(
                [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch]
            )
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Dump JSONL format (for programmatic access)
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )
            
            # Dump human-readable format (for easy inspection)
            readable_dir = os.path.join(val_data_dir, "readable")
            self._dump_responses_readable(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=readable_dir,
                prefix="validation",
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if sample_scores:
            metric_dict["val-core/all_data_sources/reward/mean"] = float(np.mean(sample_scores))

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        # Optional: upload checkpoint dir to wandb as an artifact
        if self.config.trainer.get("wandb_upload_checkpoints", False) and hasattr(self, "tracking"):
            art_name = self.config.trainer.get(
                "wandb_checkpoint_artifact_name",
                f"{self.config.trainer.experiment_name}-checkpoint",
            )
            self.tracking.log_artifact(
                path=local_global_step_folder,
                name=art_name,
                artifact_type=self.config.trainer.get("wandb_checkpoint_artifact_type", "model"),
                aliases=["latest", f"step-{self.global_steps}"],
                metadata={"global_step": int(self.global_steps)},
            )

        # Optional: merge FSDP checkpoint and upload to Hugging Face Hub
        if self.config.trainer.get("hf_upload_checkpoints", False):
            hf_upload_path = self.config.trainer.get("hf_upload_path")
            if not hf_upload_path:
                raise ValueError(
                    "hf_upload_path must be set when hf_upload_checkpoints is True. "
                    "Use a template like 'username/{experiment_name}-step-{global_steps}'"
                )
            hf_upload_path = (
                str(hf_upload_path)
                .replace("{experiment_name}", self.config.trainer.experiment_name)
                .replace("{global_steps}", str(self.global_steps))
            )
            merge_temp_dir = os.path.join(local_global_step_folder, "_hf_merge_temp")
            local_mkdir_safe(merge_temp_dir)
            trust_remote_code = self.config.actor_rollout_ref.model.get("trust_remote_code", False)
            cmd = [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                actor_local_path,
                "--target_dir",
                merge_temp_dir,
                "--hf_upload_path",
                hf_upload_path,
            ]
            if self.config.trainer.get("hf_upload_private", False):
                cmd.append("--private")
            if trust_remote_code:
                cmd.append("--trust-remote-code")
            print(f"Uploading checkpoint to Hugging Face: {hf_upload_path}")
            try:
                subprocess.run(cmd, check=True)
                print(f"Successfully uploaded checkpoint to https://huggingface.co/{hf_upload_path}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Hugging Face upload failed: {e}")
            finally:
                if os.path.exists(merge_temp_dir):
                    try:
                        shutil.rmtree(merge_temp_dir)
                    except OSError as cleanup_err:
                        print(f"Warning: Failed to cleanup merge temp dir: {cleanup_err}")

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute rollout importance sampling weights and mismatch metrics, conditionally add weights to batch.

        This method computes IS weights to correct for distribution mismatch between
        rollout policy and training policy. It always computes metrics when enabled, but
        only adds weights to batch if algorithm.rollout_is is True.

        Args:
            batch: DataProto containing old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics) where:
                - updated_batch: Batch with rollout_is_weights added (if rollout_is=True)
                - metrics: Dictionary of IS and mismatch metrics (all with mismatch/ prefix)
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch
        if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
            rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.rollout_is_threshold_lower,
                rollout_is_veto_threshold=self.config.algorithm.rollout_is_veto_threshold,
            )

            # Control: Should we apply weights to policy loss?
            # True = add weights to batch (actor will apply them)
            # False = don't add weights (metrics only, no loss modification)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights to batch for distribution to workers
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # Keep a reference so we can log artifacts (e.g., checkpoints/rollouts)
        self.tracking = logger

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                # Inject personalities if enabled
                # NOTE: We need to copy data_source from batch to gen_batch for personality injection
                personality_ids = None
                if self.use_personality_diversity and self.personalities is not None:
                    with marked_timer("inject_personality", timing_raw, color="cyan"):
                        # Copy data_source and extra_info from batch to gen_batch for personality injection
                        rollout_n = self.config.actor_rollout_ref.rollout.n
                        
                        if "data_source" in batch.non_tensor_batch:
                            original_data_source = batch.non_tensor_batch["data_source"]
                            repeated_data_source = np.repeat(original_data_source, rollout_n)
                            gen_batch.non_tensor_batch["data_source"] = repeated_data_source
                            print(f"[Personality] Copied data_source to gen_batch (size: {len(repeated_data_source)})")
                        
                        if "extra_info" in batch.non_tensor_batch:
                            original_extra_info = batch.non_tensor_batch["extra_info"]
                            repeated_extra_info = np.repeat(original_extra_info, rollout_n)
                            gen_batch.non_tensor_batch["extra_info"] = repeated_extra_info
                            print(f"[Personality] Copied extra_info to gen_batch (size: {len(repeated_extra_info)})")
                        
                        if "raw_prompt" in batch.non_tensor_batch:
                            original_raw_prompt = batch.non_tensor_batch["raw_prompt"]
                            repeated_raw_prompt = np.repeat(original_raw_prompt, rollout_n)
                            gen_batch.non_tensor_batch["raw_prompt"] = repeated_raw_prompt
                            print(f"[Personality] Copied raw_prompt to gen_batch (size: {len(repeated_raw_prompt)})")
                        
                        if not any(k in batch.non_tensor_batch for k in ["extra_info", "raw_prompt"]):
                            print(f"[Personality] Warning: Neither 'extra_info' nor 'raw_prompt' found. Keys: {list(batch.non_tensor_batch.keys())}")
                        
                        gen_batch = inject_personality_to_batch(
                            gen_batch,
                            self.personalities,
                            self.tokenizer,
                            self.processor,
                            inject_mode=self.config.algorithm.get("personality_inject_mode", "system"),
                            apply_chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}),
                        )
                        
                        # Save personality_ids to add to final batch later
                        if "personality_id" in gen_batch.non_tensor_batch:
                            personality_ids = gen_batch.non_tensor_batch["personality_id"].copy()
                            print(f"[Personality] Saved personality_ids for propagation (size: {len(personality_ids)}, unique: {np.unique(personality_ids)})")
                        
                        # Log sample prompts to verify personality injection
                        if self.global_steps <= 3:  # Only log first few steps
                            sample_prompts = self.tokenizer.batch_decode(
                                gen_batch.non_tensor_batch["raw_prompt_ids"][:3], 
                                skip_special_tokens=False
                            )
                            print(f"\n[Personality Injection Check - Step {self.global_steps}]")
                            for i, prompt in enumerate(sample_prompts):
                                # Skip padding tokens to show actual content
                                # Qwen uses <|endoftext|> as pad token
                                clean_prompt = prompt.replace("<|endoftext|>", "").strip()
                                print(f"  Sample {i} (personality_id={personality_ids[i] if personality_ids is not None else 'N/A'}):")
                                print(f"    Prompt (first 800 chars, padding removed): {clean_prompt[:800]}...")
                                print()
                
                gen_batch_output=gen_batch  # initialize
                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        
                        # Log sample prompts to verify personality injection
                        if self.global_steps <= 3:  # Only log first few steps
                            sample_responses = self.tokenizer.batch_decode(
                                gen_batch_output.batch["responses"][:3], 
                                skip_special_tokens=False
                            )
                            print(f"\n[Personality Response Check - Step {self.global_steps}]")
                            for i, response in enumerate(sample_responses):
                                # Skip padding tokens to show actual content
                                # Qwen uses <|endoftext|> as pad token
                                clean_response = response.replace("<|endoftext|>", "").strip()
                                print(f"  Sample {i} (personality_id={personality_ids[i] if personality_ids is not None else 'N/A'}):")
                                print(f"    Response (first 800 chars, padding removed): {clean_response}...")
                                print()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    
                    # Propagate personality_ids to the final batch for diversity reward computation
                    if personality_ids is not None:
                        batch.non_tensor_batch["personality_id"] = personality_ids
                        print(f"[Personality] Propagated personality_ids to final batch (size: {len(personality_ids)})")

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score (if using reward model)
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            print(f"[Reward Check] Using reward model (use_rm=True) at step {self.global_steps}")
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                            metrics["reward/reward_source"] = "reward_model"
                        else:
                            metrics["reward/reward_source"] = "rule_based_reward_fn"

                        reward_future = None
                        # compute rule-based reward (e.g., ground truth comparison for math)
                        # Note: For math datasets with ground truth, use_rm should be False and only reward_fn is used
                        if self.reward_fn is not None:
                            # Log which reward function is being used
                            reward_fn_name = getattr(self.reward_fn, '__name__', 'unknown')
                            reward_fn_class = type(self.reward_fn).__name__
                            print(f"[Reward Check] Using reward_fn at step {self.global_steps}: {reward_fn_class}.{reward_fn_name}")
                            
                            # Check if it's using math reward (regex heuristic)
                            if hasattr(self.reward_fn, 'compute_score'):
                                compute_score_fn = self.reward_fn.compute_score
                                compute_score_name = getattr(compute_score_fn, '__name__', 'unknown')
                                if 'math' in compute_score_name.lower() or 'compute_math_reward' in str(compute_score_fn):
                                    print(f"[Reward Check] Math reward (regex heuristic) detected: {compute_score_name}")
                                    metrics["reward/math_reward_used"] = 1.0
                                else:
                                    metrics["reward/math_reward_used"] = 0.0
                            
                            if self.config.reward_model.launch_reward_fn_async:
                                assert self.reward_fn_actor is not None, "Reward actor must be initialized for async rewards."
                                reward_future = self.reward_fn_actor.compute.remote(batch)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        elif not self.use_rm:
                            raise ValueError("Either reward_fn or use_rm must be enabled for reward computation")

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(reward_future)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            # Log mean breakdowns from custom reward for easy monitoring (e.g., wandb)
                            for key in reward_extra_infos_dict.keys():
                                vals = reward_extra_infos_dict.get(key, None)
                                if vals:
                                    metrics[f"reward/{key}_mean"] = float(np.mean(vals))

                        # Compute diversity reward if personality diversity is enabled
                        # I think this is not needed anymore since we are using the custom reward function.
                        if self.use_personality_diversity:
                            with marked_timer("diversity_reward", timing_raw, color="magenta"):
                                # Debug: Check if personality_id is in batch
                                has_personality_id = "personality_id" in batch.non_tensor_batch
                                has_uid = "uid" in batch.non_tensor_batch
                                has_response_mask = "response_mask" in batch.batch
                                has_token_level_scores = "token_level_scores" in batch.batch
                                
                                if self.global_steps <= 3:  # Log debug info for first few steps
                                    print(f"\n[Diversity Reward Debug - Step {self.global_steps}]")
                                    print(f"  has personality_id: {has_personality_id}")
                                    print(f"  has uid: {has_uid}")
                                    print(f"  has response_mask: {has_response_mask}")
                                    print(f"  has token_level_scores: {has_token_level_scores}")
                                    if has_personality_id:
                                        pid = batch.non_tensor_batch["personality_id"]
                                        print(f"  personality_id shape: {pid.shape if hasattr(pid, 'shape') else len(pid)}")
                                        print(f"  personality_id unique values: {np.unique(pid)}")
                                    if has_uid:
                                        uid = batch.non_tensor_batch["uid"]
                                        print(f"  uid sample: {uid[:5] if len(uid) > 5 else uid}")
                                
                                diversity_rewards, diversity_metrics = compute_diversity_reward(
                                    batch,
                                    self.tokenizer,
                                    metric=self.config.algorithm.get("diversity_metric", "token_overlap"),
                                    diversity_weight=self.config.algorithm.get("diversity_reward_weight", 0.1),
                                )
                                
                                # Log diversity reward statistics
                                diversity_sum = diversity_rewards.sum().item()
                                diversity_nonzero = (diversity_rewards != 0).sum().item()
                                if self.global_steps <= 3:
                                    print(f"  diversity_rewards sum: {diversity_sum:.6f}")
                                    print(f"  diversity_rewards nonzero count: {diversity_nonzero}")
                                    print(f"  diversity_metrics: {diversity_metrics}")
                                
                                # Add diversity reward to token_level_scores
                                batch.batch["token_level_scores"] = batch.batch["token_level_scores"] + diversity_rewards
                                metrics.update(diversity_metrics)
                                
                                # Also log the actual diversity reward sum for monitoring
                                metrics["diversity/reward_sum"] = diversity_sum
                                metrics["diversity/nonzero_count"] = diversity_nonzero
                                
                                # Compute and save per-personality metrics
                                with marked_timer("personality_analysis", timing_raw, color="cyan"):
                                    personality_metrics = self._compute_personality_metrics(
                                        batch, 
                                        batch.batch["token_level_scores"],
                                        diversity_metrics,
                                    )
                                    
                                    # Add personality metrics to wandb (excluding internal data)
                                    for k, v in personality_metrics.items():
                                        if not k.startswith("_"):
                                            metrics[k] = v
                                    
                                    # Save to JSON file
                                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                                    if rollout_data_dir:
                                        self._save_personality_analysis(personality_metrics, rollout_data_dir)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate (periodic evaluation)
                eval_freq = self.config.trainer.get("eval_freq", None)  # Evaluation frequency (in steps)
                test_freq = self.config.trainer.get("test_freq", 0)  # Legacy validation frequency
                
                # Run evaluation if:
                # 1. eval_freq is set and current step is a multiple of eval_freq, OR
                # 2. test_freq is set (legacy) and current step is a multiple of test_freq, OR
                # 3. It's the last step
                should_eval = False
                if eval_freq is not None and eval_freq > 0:
                    should_eval = (is_last_step or self.global_steps % eval_freq == 0)
                elif test_freq > 0:
                    should_eval = (is_last_step or self.global_steps % test_freq == 0)
                
                if self.val_reward_fn is not None and should_eval:
                    with marked_timer("testing", timing_raw, color="green"):
                        print(f"\n{'='*80}")
                        print(f"Running evaluation at step {self.global_steps}")
                        print(f"{'='*80}")
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                        
                        # Log key evaluation metrics
                        if val_metrics:
                            print(f"\nEvaluation Results at Step {self.global_steps}:")
                            for key, value in sorted(val_metrics.items()):
                                if any(keyword in key.lower() for keyword in ['acc', 'reward', 'mean', 'best']):
                                    print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
                            print(f"{'='*80}\n")
                        
                        metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
