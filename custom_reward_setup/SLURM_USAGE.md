# Running Custom Reward Function Training on Stanford SLURM

## Quick Start

### 1. Modify the SLURM Script

Edit `run_custom_reward_training.slurm` and update:

```bash
# Line 8: Replace with your Stanford account
#SBATCH --account=your-account

# Line 19: Verify your work directory path
VERL_WORKDIR=/hai/scratch/jihaoliu/verl_collaborative

# Line 25-26: Update dataset paths
TRAIN_FILES="${HOME}/data/gsm8k/train.parquet"
VAL_FILES="${HOME}/data/gsm8k/test.parquet"
```

### 2. Test First (Recommended)

Before running full training, test with the smaller script:

```bash
sbatch verl_collaboration/run_custom_reward_test.slurm
```

Monitor the job:
```bash
squeue -u $USER
tail -f slurm-test-*.out
```

### 3. Run Full Training

```bash
sbatch verl_collaboration/run_custom_reward_training.slurm
```

Monitor the job:
```bash
squeue -u $USER
tail -f slurm-verl-custom-reward-*.out
```

## Configuration Options

### GPU Configuration

**Maximum 4 GPUs per job** (as per requirement):

```bash
#SBATCH --gres=gpu:4        # Maximum 4 GPUs
#SBATCH --mem=300G          # Adjusted for 4 GPUs
```

Note: Batch sizes have been adjusted accordingly for 4 GPU training.

### Model Selection

In the script, change:
```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct   # Smaller, faster
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct   # Larger, better quality
```

### Batch Sizes

Batch sizes are configured for 4 GPUs. Current settings:
```bash
TRAIN_BATCH_SIZE=128              # Total batch size (adjusted for 4 GPUs)
PPO_MINI_BATCH_SIZE=32            # Mini batch size (adjusted for 4 GPUs)
PPO_MICRO_BATCH_SIZE_PER_GPU=2    # Per-GPU micro batch size (reduced to avoid OOM)
N_RESP_PER_PROMPT=4               # Responses per prompt (reduced for 4 GPUs)
```

If you get OOM errors, further reduce:
```bash
PPO_MICRO_BATCH_SIZE_PER_GPU=1   # Further decrease if OOM errors occur
TRAIN_BATCH_SIZE=64              # Reduce total batch size
N_RESP_PER_PROMPT=2              # Reduce responses per prompt
```

### Reward Manager

For **distinctiveness support** (recommended):
```bash
REWARD_MANAGER=batch
```

For **single-item processing** (no distinctiveness):
```bash
REWARD_MANAGER=naive
```

### Reward Function Weights

Customize reward components:
```bash
MATH_WEIGHT=1.0              # Weight for math correctness
TEXT_WEIGHT=1.0              # Weight for text quality
DISTINCTIVENESS_WEIGHT=0.3   # Weight for diversity (only works with batch manager)
```

## Common Issues

### Out of Memory (OOM)

1. Reduce batch sizes:
   ```bash
   PPO_MICRO_BATCH_SIZE_PER_GPU=2
   TRAIN_BATCH_SIZE=256
   ```

2. Enable parameter offloading:
   ```bash
   actor_rollout_ref.ref.fsdp_config.param_offload=true
   ```

3. Reduce sequence lengths:
   ```bash
   MAX_PROMPT_LENGTH=512
   MAX_RESPONSE_LENGTH=1024
   ```

### Tulu Model Loading Issues

If Tulu model fails to load:
- The function will automatically fall back to heuristic scoring
- Check logs for warnings about Tulu model initialization
- Ensure you have sufficient GPU memory (~14GB for Tulu-2-7B)

### SBERT Model Loading Issues

If SBERT model fails to load:
- Distinctiveness will be disabled (returns 0.0)
- Check logs for SBERT initialization warnings
- SBERT is lightweight and should load easily

## Multi-Node Training

For multi-node training (still max 4 GPUs per node), update:

```bash
#SBATCH --nodes=2           # Number of nodes
#SBATCH --gres=gpu:4        # GPUs per node (max 4)
```

The script automatically detects `SLURM_JOB_NUM_NODES` and configures Ray accordingly.

**Note**: With 4 GPUs per node limit, multi-node training will use 4 GPUs per node.

## Monitoring Training

### Check Logs
```bash
# View output
tail -f slurm-verl-custom-reward-*.out

# View errors
tail -f slurm-verl-custom-reward-*.err

# Search for reward information
grep "reward" slurm-verl-custom-reward-*.out
```

### Check WandB (if enabled)

The script logs to WandB. Check your project dashboard:
- Project: `verl-custom-reward`
- Experiment: `hybrid-reward-YYYYMMDD-HHMMSS`

### Check Checkpoints

Checkpoints are saved to:
```bash
${VERL_WORKDIR}/checkpoints/${EXP_NAME}/
```

## Stopping Training

```bash
# Find job ID
squeue -u $USER

# Cancel job
scancel <job_id>
```

## Resuming Training

The script uses `trainer.resume_mode=auto`, so it will automatically resume from the latest checkpoint if training is interrupted.

To manually resume:
```bash
sbatch verl_collaboration/run_custom_reward_training.slurm \
    trainer.resume_mode=auto \
    trainer.default_local_dir="/path/to/checkpoint/dir"
```

## Stanford-Specific Notes

1. **Partition**: Update `#SBATCH --partition=gpu` if your cluster uses different partition names
2. **Account**: Always set `#SBATCH --account=your-account`
3. **Time Limits**: Adjust `--time` based on your cluster's limits
4. **Conda Environment**: Ensure `conda activate verl` matches your environment name

## Troubleshooting

### Job Pending

Check why:
```bash
squeue -j <job_id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

Common reasons:
- Insufficient resources (reduce GPU/memory requests)
- Partition limits (check partition availability)
- Account limits (verify account has GPU access)

### Import Errors

Ensure you're in the correct directory:
```bash
cd /hai/scratch/jihaoliu/verl_collaborative
```

Verify Python path:
```bash
which python3
python3 -c "import verl; print(verl.__file__)"
```

### Custom Reward Function Not Found

Verify path:
```bash
ls -la /hai/scratch/jihaoliu/verl_collaborative/verl_collaboration/custom_reward_function.py
```

Check that the path in the script matches your actual file location.

