# Base directory for verl_collaborative
echo $HOME
VERL_WORKDIR=$HOME

CONFIG_NAME=no_thinking_classifier
CONFIG_PATH=${CONFIG_NAME}.yaml
CONFIG_DIR="$(dirname "$CONFIG_PATH")"

# Scratch directory for large files (checkpoints, rollout data)
# This has much more disk space than $HOME
SCRATCH_DIR=$HOME/verl_collaboration/scratch

# Path to your custom reward function
CUSTOM_REWARD_FN_PATH="${VERL_WORKDIR}/verl_collaboration/custom_reward_setup/infinite_chat_reward_function.py"

# Model configuration — Instruct model instead of Base
MODEL_FAMILY="Qwen"
MODEL_NAME="Qwen3-8B"
MODEL_PATH="${MODEL_FAMILY}/${MODEL_NAME}"

# Experiment configuration
export WANDB_ENTITY="MARL_collab"
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.wandb_api_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_api_key")
fi

DATASET_NAME="tulu3_infinite-chats-taxonomy_mix_10k_no_thinking"

PROJECT_NAME="verl-${DATASET_NAME}-training"
# Fixed name so checkpoint-resume works across preemptions
EXP_NAME=$CONFIG_NAME

CHECKPOINT_DIR="${SCRATCH_DIR}/verl_checkpoints/${EXP_NAME}"
ROLLOUT_DATA_DIR="${SCRATCH_DIR}/verl_rollout_data/${EXP_NAME}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${ROLLOUT_DATA_DIR}"
echo "[INFO] Checkpoint directory: ${CHECKPOINT_DIR}"
echo "[INFO] Rollout data directory: ${ROLLOUT_DATA_DIR}"

# Unset ROCm/HIP variables to avoid conflicts with CUDA
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES


# Disable Ray dashboard to avoid OpenTelemetry crash
export RAY_INCLUDE_DASHBOARD=false
export RAY_USAGE_STATS_ENABLED=0
export RAY_DEDUP_LOGS=0
export OTEL_SDK_DISABLED=true


# Persist wandb run ID so resumed jobs continue the same wandb run
WANDB_ID_FILE="${CHECKPOINT_DIR}/.wandb_run_id"
if [ -f "${WANDB_ID_FILE}" ]; then
    export WANDB_RUN_ID=$(cat "${WANDB_ID_FILE}")
    export WANDB_RESUME="must"
    echo "[INFO] Resuming wandb run: ${WANDB_RUN_ID}"
fi

python -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    ++algorithm.adv_estimator=grpo \
    ++data.train_batch_size=32 \
    ++data.data_sampling_seed=42 \
    ++data.val_batch_size=32 \
    ++data.max_prompt_length=1024 \
    ++data.max_response_length=2048 \
    ++actor_rollout_ref.rollout.prompt_length=1024 \
    ++actor_rollout_ref.rollout.response_length=2048 \
    ++actor_rollout_ref.rollout.max_model_len=4096 \
    ++data.filter_overlong_prompts=true \
    ++data.truncation='error' \
    ++actor_rollout_ref.model.path="${MODEL_PATH}" \
    ++actor_rollout_ref.actor.optim.lr=1e-6 \
    ++actor_rollout_ref.model.use_remove_padding=False \
    ++actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    ++actor_rollout_ref.actor.use_kl_loss=True \
    ++actor_rollout_ref.actor.kl_loss_coef=0.01 \
    ++actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    ++actor_rollout_ref.actor.entropy_coeff=0 \
    ++actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.actor.fsdp_config.param_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    ++actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    ++actor_rollout_ref.rollout.name=vllm \
    ++actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    ++actor_rollout_ref.rollout.n=6 \
    ++actor_rollout_ref.rollout.val_kwargs.n=6 \
    ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    ++actor_rollout_ref.ref.fsdp_config.param_offload=True \
    ++reward_model.enable=false \
    ++reward_model.reward_manager=batch \
    ++reward_model.launch_reward_fn_async=false \
    ++trainer.critic_warmup=0 \
    ++custom_reward_function.path="${CUSTOM_REWARD_FN_PATH}" \
    ++custom_reward_function.name="hybrid_reward_function" \
    ++custom_reward_function.reward_kwargs.math_weight=0 \
    ++custom_reward_function.reward_kwargs.thresholding_base_reward=0.5 \
    ++custom_reward_function.reward_kwargs.length_penalty_weight=0.01 \
    ++custom_reward_function.reward_kwargs.language_penalty_weight=0.1 \
    ++algorithm.use_generated_personalities=false \
    ++algorithm.diversity_reward_weight=0 \
    ++trainer.logger='["console","wandb"]' \
    ++trainer.project_name="${PROJECT_NAME}" \
    ++trainer.experiment_name="${EXP_NAME}" \
    ++trainer.default_local_dir="${CHECKPOINT_DIR}" \
    ++trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    ++trainer.wandb_upload_rollout_data=false \
    ++trainer.wandb_upload_checkpoints=false \
    ++trainer.n_gpus_per_node=3 \
    ++trainer.nnodes=1 \
    ++trainer.val_before_train=true \
    ++trainer.eval_freq=100 \
    ++trainer.eval_sample_num=512 \
    ++trainer.test_freq=0 \
    ++trainer.save_freq=25 \
    ++trainer.total_epochs=2 \
    ++trainer.resume_mode=auto \
    ++trainer.max_actor_ckpt_to_keep=3 \
    ++trainer.max_critic_ckpt_to_keep=3 \
    ++ray_kwargs.ray_init.include_dashboard=false \
