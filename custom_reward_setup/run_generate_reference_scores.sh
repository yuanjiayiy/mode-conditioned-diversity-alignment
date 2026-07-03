#!/usr/bin/env bash
#SBATCH --job-name=preprocess-infinite-chats                     
#SBATCH --nodes=1                          
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=16                 # CPU cores per task (adjusted for 2 GPUs)
#SBATCH --gres=gpu:2                       # Number of GPUs per node
#SBATCH --mem=200G                          # Memory per node (adjusted for 2 GPUs)
#SBATCH --time=1-00:00:00                  # 3 days - adjust batch sizes/epochs to fit
#SBATCH --output=logs/preprocess-%x-%j.out  # Output file in logs directory
#SBATCH --error=logs/preprocess-%x-%j.err   # Error file in logs directory


set -xeuo pipefail

# Optionally, download and preprocess the dataset
# python ${VERL_WORKDIR}/examples/data_preprocess/tulu3_infinite_chats_taxonomy_mix.py

# Configuration
VERL_WORKDIR=${VERL_WORKDIR:-$HOME/verl_collaboration}
SCRATCH_DIR=${SCRATCH_DIR:-$HOME/scratch}
LOCAL_SAVE_DIR="${VERL_WORKDIR}/data/tulu3_infinite-chats-taxonomy_mix_10k_w_thinking"
HF_DATASET_NAME="tulu3_infinite-chats-taxonomy_mix_10k_w_thinking"

# If OOM in Phase 2: run with --phase1_only first, then run again (skips Phase 1, loads Skyreward only)
VISIBLE_CUDA_DEVICES=0,1,2,3 python -m examples.data_preprocess.generate_reference_scores \
    --input_dir $LOCAL_SAVE_DIR \
    --base_model_path "Qwen/Qwen3-8B" \
    --max_tokens_for_thinking 0 \
    --n_samples 5 \
    --skyreward_batch_size 8 \
    --push_to_hub \
    --splits train \
    --dataset_name ${HF_DATASET_NAME}