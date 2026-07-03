# Custom Reward Function Architecture Guide

## Overview

This document explains how custom reward functions work in the VERL PPO training system.

## File Architecture

```
verl_collaborative/
├── custom_reward_function.py          # ✅ YOUR CUSTOM FILE (CREATE THIS)
├── config_example_custom_reward.yaml  # ✅ YOUR CONFIG (MODIFY THIS)
│
└── verl_collaboration/verl/trainer/
    ├── main_ppo.py                    # ❌ DO NOT MODIFY
    │   └── TaskRunner.run() [lines 225-317]
    │       └── loads reward_fn [lines 282-287]
    │
    └── ppo/
        ├── reward.py                  # ❌ DO NOT MODIFY
        │   ├── get_custom_reward_fn() [lines 42-93]
        │   │   └── Dynamically loads YOUR file
        │   ├── load_reward_manager() [lines 96-152]
        │   │   └── Orchestrates reward loading
        │   └── compute_reward() [lines 155-173]
        │       └── Calls your function during training
        │
        └── ray_trainer.py             # ❌ DO NOT MODIFY
            └── Uses reward_fn throughout training
```

## Data Flow During Training

```
1. Training Start
   │
   ├─► main_ppo.py::run_ppo() initializes Ray
   │
   └─► TaskRunner.run() executes
       │
       ├─► Line 282: load_reward_manager(config, tokenizer, ...)
       │   │
       │   └─► reward.py::load_reward_manager()
       │       │
       │       ├─► Line 114: compute_score = get_custom_reward_fn(config)
       │       │   │
       │       │   └─► reward.py::get_custom_reward_fn()
       │       │       │
       │       │       ├─► Line 62: Read config.custom_reward_function.path
       │       │       ├─► Line 67: Read config.custom_reward_function.name
       │       │       ├─► Lines 70-81: Import your Python file dynamically
       │       │       ├─► Line 89: Get your function by name
       │       │       ├─► Line 91: Extract reward_kwargs from config
       │       │       └─► Line 93: Return wrapped function with kwargs
       │       │
       │       ├─► Line 126: Get reward_manager_cls (e.g., NaiveRewardManager)
       │       └─► Line 146: Instantiate manager with your function
       │
       └─► Line 306: Pass reward_fn to RayPPOTrainer
           │
           └─► During training iterations:
               │
               └─► ray_trainer.py calls reward_fn(data)
                   │
                   └─► reward.py::compute_reward(data, reward_fn)
                       │
                       └─► YOUR FUNCTION: hybrid_reward_function(data, **kwargs)
                           │
                           ├─► Receives DataProto with batch data
                           ├─► Processes each item in batch
                           ├─► Computes rewards based on your logic
                           └─► Returns {
                                 'reward_tensor': torch.Tensor,
                                 'reward_extra_info': dict
                               }
```

## DataProto Structure

When your function is called, it receives a `DataProto` object with this structure:

```python
data: DataProto
    └── data.batch: List[Dict]
        └── data.batch[i]: Dict containing:
            ├── 'prompt': str           # Input prompt
            ├── 'response': str         # Generated response
            ├── 'input_ids': Tensor     # Tokenized input
            ├── 'attention_mask': Tensor
            ├── 'data_source': str      # Task type: 'math', 'text', 'code'
            ├── 'ground_truth': str     # (Optional) Correct answer
            └── ... other fields depending on dataset
```

## Your Custom Function Signature

```python
def hybrid_reward_function(
    data: DataProto,
    # These come from config.custom_reward_function.reward_kwargs:
    math_weight: float = 1.0,
    text_weight: float = 1.0,
    distinctiveness_weight: float = 0.3,
    **kwargs
) -> Dict[str, Any]:
    """
    Returns:
        dict: {
            'reward_tensor': torch.Tensor of shape [batch_size],
            'reward_extra_info': dict with additional metrics
        }
    
    Or just return torch.Tensor for backward compatibility.
    """
```

## Configuration Flow

```
ppo_trainer.yaml
    │
    ├─► custom_reward_function:
    │   ├─► path: "/absolute/path/to/custom_reward_function.py"
    │   ├─► name: "hybrid_reward_function"
    │   └─► reward_kwargs:
    │       ├─► math_weight: 1.0
    │       ├─► text_weight: 1.0
    │       └─► distinctiveness_weight: 0.3
    │
    └─► reward_model:
        ├─► enable: False  # Use custom function only
        └─► reward_manager: "naive"  # How to batch/process rewards
```

## Reward Manager Types

The `reward_manager` field determines how your function is called:

- **`naive`**: Calls your function once per batch (default)
- **`batch`**: Better batching for large-scale processing
- **`prime`**: For PRIME algorithm with advantage estimation
- **`dapo`**: For DAPO algorithm

## How to Use Different Data Types

Your function receives `data_source` field to distinguish task types:

```python
data_source = item.get('data_source', 'text')

if data_source in ['math', 'gsm8k', 'MATH']:
    # Use math reward logic
    reward = compute_math_reward(response, ground_truth)
    
elif data_source in ['text', 'chat', 'instruction']:
    # Use text quality logic
    reward = compute_text_quality_reward(response, prompt)
    
elif data_source in ['code', 'programming']:
    # Use code execution logic
    reward = compute_code_reward(response, test_cases)
```

## Complete Setup Steps

### Step 1: Create Your Custom Reward File

```bash
# File: /hai/scratch/jihaoliu/verl_collaborative/custom_reward_function.py
# Already created! See the file for implementation.
```

### Step 2: Update Your Config

Edit your training config (e.g., `config/ppo_trainer.yaml`):

```yaml
custom_reward_function:
  path: "/hai/scratch/jihaoliu/verl_collaborative/custom_reward_function.py"
  name: "hybrid_reward_function"
  reward_kwargs:
    math_weight: 1.0
    text_weight: 1.0
    distinctiveness_weight: 0.3
```

### Step 3: Ensure Your Dataset Has Required Fields

Your dataset should provide items with:
- `prompt`: The input
- `response`: The generated output (added by model)
- `data_source`: Task type ('math', 'text', etc.)
- `ground_truth`: (Optional) correct answer for evaluation

### Step 4: Run Training

```bash
python verl_collaboration/verl/trainer/main_ppo.py \
    --config-name your_config.yaml
```

## Testing Your Reward Function

Before running full training, test your function:

```python
# test_reward.py
import torch
from verl import DataProto
from custom_reward_function import hybrid_reward_function

# Create mock data
mock_data = DataProto()
mock_data.batch = [
    {
        'prompt': 'What is 2+2?',
        'response': 'The answer is \\boxed{4}',
        'data_source': 'math',
        'ground_truth': '4'
    },
    {
        'prompt': 'Explain photosynthesis',
        'response': 'Photosynthesis is the process by which plants...',
        'data_source': 'text',
    }
]

# Test your function
result = hybrid_reward_function(
    mock_data,
    math_weight=1.0,
    text_weight=1.0,
    distinctiveness_weight=0.3
)

print("Rewards:", result['reward_tensor'])
print("Extra info:", result['reward_extra_info'])
```

## Advanced: Combining with Reward Model

If you want to use BOTH your custom function AND a reward model:

```yaml
custom_reward_function:
  path: "/path/to/custom_reward_function.py"
  name: "hybrid_reward_function"

reward_model:
  enable: True  # Enable reward model
  model:
    path: ~/models/my-reward-model
  # Your custom function will be combined with model scores
```

The system will:
1. Call your custom function
2. Call the reward model
3. Combine both scores (implementation in reward_manager)

## Debugging Tips

1. **Check function is loaded**:
   Look for this line in logs:
   ```
   using customized reward function 'hybrid_reward_function' from '...'
   ```

2. **Print from your function**:
   Add debug prints in your custom function:
   ```python
   print(f"Processing batch of size {len(data.batch)}")
   ```

3. **Check reward_extra_info**:
   Your extra info will be logged during training

4. **Verify data structure**:
   Print `data.batch[0]` to see what fields are available

## Common Issues

### Issue: Function not found
```
AttributeError: Reward function 'my_function' not found
```
**Solution**: Check `name` in config matches function name exactly

### Issue: Import errors in custom file
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution**: Make sure all imports in your custom file are available in the environment

### Issue: Reward tensor shape mismatch
```
RuntimeError: Expected tensor of shape [batch_size]
```
**Solution**: Ensure you return `torch.Tensor` of shape `[len(data.batch)]`

## Summary

✅ **CREATE**: `custom_reward_function.py` with your reward logic  
✅ **MODIFY**: `config/ppo_trainer.yaml` to point to your function  
❌ **DON'T MODIFY**: Any files in `verl_collaboration/verl/trainer/`  

The system automatically:
- Discovers your function via config
- Loads it dynamically at runtime
- Passes it the data during training
- Collects rewards and updates the policy

Everything is config-driven and modular! 🎉


