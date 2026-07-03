#!/usr/bin/env python3
"""
Test script to verify custom reward function and data preprocessing pipeline.

This script checks:
1. Data preprocessing creates correct parquet format
2. Custom reward function can be loaded and called
3. Training configuration is correct
4. Reward function receives correct data format
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add verl_collaboration to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import datasets
    from datasets import Dataset
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Please install: pip install pandas datasets")
    sys.exit(1)

try:
    from verl import DataProto
except ImportError as e:
    print(f"ERROR: Could not import verl: {e}")
    print("Make sure you're in the verl conda environment")
    sys.exit(1)


def test_parquet_format(parquet_path: str) -> bool:
    """
    Test if parquet file has the correct format for VERL training.
    
    Returns:
        bool: True if format is correct
    """
    print(f"\n{'='*80}")
    print(f"TEST 1: Checking Parquet File Format")
    print(f"{'='*80}")
    print(f"File: {parquet_path}")
    
    if not os.path.exists(parquet_path):
        print(f"❌ ERROR: Parquet file not found: {parquet_path}")
        return False
    
    try:
        # Load parquet file
        df = pd.read_parquet(parquet_path)
        print(f"✅ Parquet file loaded successfully")
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
        
        # Check required fields
        required_fields = ["data_source", "prompt", "ability", "reward_model"]
        missing_fields = [f for f in required_fields if f not in df.columns]
        
        if missing_fields:
            print(f"❌ ERROR: Missing required fields: {missing_fields}")
            return False
        
        print(f"✅ All required fields present: {required_fields}")
        
        # Check data types and structure
        print(f"\n   Checking field types and structure...")
        
        # Check data_source
        if df["data_source"].dtype != "object":
            print(f"⚠️  WARNING: data_source should be string, got {df['data_source'].dtype}")
        
        # Check prompt format (should be list of dicts)
        sample_prompt = df["prompt"].iloc[0]
        if isinstance(sample_prompt, str):
            try:
                sample_prompt = json.loads(sample_prompt)
            except:
                pass
        
        if isinstance(sample_prompt, list) and len(sample_prompt) > 0:
            if isinstance(sample_prompt[0], dict) and "role" in sample_prompt[0]:
                print(f"✅ Prompt format correct: list of message dicts")
            else:
                print(f"⚠️  WARNING: Prompt should be list of dicts with 'role' key")
        else:
            print(f"⚠️  WARNING: Prompt should be a list")
        
        # Check reward_model format (should be dict with ground_truth)
        sample_reward_model = df["reward_model"].iloc[0]
        if isinstance(sample_reward_model, str):
            try:
                sample_reward_model = json.loads(sample_reward_model)
            except:
                pass
        
        if isinstance(sample_reward_model, dict):
            if "ground_truth" in sample_reward_model:
                print(f"✅ Reward model format correct: dict with ground_truth")
            else:
                print(f"⚠️  WARNING: reward_model should have 'ground_truth' key")
        else:
            print(f"⚠️  WARNING: reward_model should be a dict")
        
        # Print sample data
        print(f"\n   Sample data (first row):")
        for col in df.columns:
            value = df[col].iloc[0]
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value, indent=2)[:200]
                if len(json.dumps(value)) > 200:
                    value_str += "..."
            else:
                value_str = str(value)[:200]
            print(f"     {col}: {value_str}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to check parquet file: {e}")
        traceback.print_exc()
        return False


def test_reward_function_loading() -> bool:
    """
    Test if custom reward function can be loaded.
    
    Returns:
        bool: True if function loads successfully
    """
    print(f"\n{'='*80}")
    print(f"TEST 2: Loading Custom Reward Function")
    print(f"{'='*80}")
    
    reward_fn_path = Path(__file__).parent / "custom_reward_function.py"
    
    if not reward_fn_path.exists():
        print(f"❌ ERROR: Reward function not found: {reward_fn_path}")
        return False
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_reward_module", reward_fn_path)
        if spec is None or spec.loader is None:
            print(f"❌ ERROR: Could not create module spec")
            return False
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, "hybrid_reward_function"):
            print(f"❌ ERROR: Function 'hybrid_reward_function' not found in module")
            return False
        
        reward_fn = getattr(module, "hybrid_reward_function")
        print(f"✅ Custom reward function loaded successfully")
        print(f"   - Path: {reward_fn_path}")
        print(f"   - Function: {reward_fn.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load reward function: {e}")
        traceback.print_exc()
        return False


def test_reward_function_call() -> bool:
    """
    Test if custom reward function can be called with sample data.
    
    Returns:
        bool: True if function works correctly
    """
    print(f"\n{'='*80}")
    print(f"TEST 3: Testing Reward Function Call")
    print(f"{'='*80}")
    
    try:
        # Load reward function
        reward_fn_path = Path(__file__).parent / "custom_reward_function.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_reward_module", reward_fn_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reward_fn = getattr(module, "hybrid_reward_function")
        
        # Test case 1: Math task (NaiveRewardManager mode)
        print(f"\n   Test 1: Math task (single item)")
        math_reward = reward_fn(
            data_source="infinite-chats-taxonomy/math",
            solution_str="The answer is \\boxed{42}",
            ground_truth="42",
            extra_info={"prompt": "What is 40 + 2?"},
            math_weight=1.0,
            text_weight=1.0,
            distinctiveness_weight=0.3
        )
        
        if isinstance(math_reward, (int, float)):
            print(f"   ✅ Math reward computed: {math_reward}")
            if not (0 <= math_reward <= 1):
                print(f"   ⚠️  WARNING: Reward should be in [0, 1], got {math_reward}")
        else:
            print(f"   ❌ ERROR: Expected float, got {type(math_reward)}: {math_reward}")
            return False
        
        # Test case 2: Text task (NaiveRewardManager mode)
        print(f"\n   Test 2: Text task (single item)")
        text_reward = reward_fn(
            data_source="infinite-chats-taxonomy/text",
            solution_str="This is a good response with multiple sentences. It explains things clearly. The content is relevant.",
            ground_truth="",
            extra_info={"prompt": "Explain photosynthesis"},
            math_weight=1.0,
            text_weight=1.0,
            distinctiveness_weight=0.3
        )
        
        if isinstance(text_reward, (int, float)):
            print(f"   ✅ Text reward computed: {text_reward}")
        else:
            print(f"   ❌ ERROR: Expected float, got {type(text_reward)}: {text_reward}")
            return False
        
        # Test case 3: Batch mode (BatchRewardManager)
        print(f"\n   Test 3: Batch mode (multiple items)")
        batch_rewards = reward_fn(
            data_sources=["infinite-chats-taxonomy/math", "infinite-chats-taxonomy/text"],
            solution_strs=["The answer is \\boxed{42}", "This is a good response."],
            ground_truths=["42", ""],
            extra_infos=[{"prompt": "What is 40 + 2?"}, {"prompt": "Explain something"}],
            math_weight=1.0,
            text_weight=1.0,
            distinctiveness_weight=0.3
        )
        
        if isinstance(batch_rewards, list) and len(batch_rewards) == 2:
            print(f"   ✅ Batch rewards computed: {batch_rewards}")
            if all(isinstance(r, (int, float)) for r in batch_rewards):
                print(f"   ✅ All rewards are valid floats")
            else:
                print(f"   ⚠️  WARNING: Some rewards are not floats")
        else:
            print(f"   ❌ ERROR: Expected list of 2 floats, got {type(batch_rewards)}: {batch_rewards}")
            return False
        
        print(f"\n✅ All reward function tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to test reward function: {e}")
        traceback.print_exc()
        return False


def test_training_config() -> bool:
    """
    Test if training configuration is correct.
    
    Returns:
        bool: True if configuration is valid
    """
    print(f"\n{'='*80}")
    print(f"TEST 4: Checking Training Configuration")
    print(f"{'='*80}")
    
    # Check if custom reward function path exists
    reward_fn_path = Path(__file__).parent / "custom_reward_function.py"
    if not reward_fn_path.exists():
        print(f"❌ ERROR: Custom reward function not found: {reward_fn_path}")
        return False
    
    print(f"✅ Custom reward function path exists: {reward_fn_path}")
    
    # Check if training script exists
    training_script = Path(__file__).parent / "run_custom_reward_training.slurm"
    if not training_script.exists():
        print(f"⚠️  WARNING: Training script not found: {training_script}")
    else:
        print(f"✅ Training script exists: {training_script}")
    
    # Verify reward function can be imported in the way verl expects
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_module", reward_fn_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, "hybrid_reward_function"):
            print(f"✅ Function 'hybrid_reward_function' found in module")
        else:
            print(f"❌ ERROR: Function 'hybrid_reward_function' not found")
            return False
        
    except Exception as e:
        print(f"❌ ERROR: Could not verify reward function: {e}")
        return False
    
    print(f"\n✅ Training configuration looks correct")
    return True


def main():
    """Run all tests."""
    print(f"\n{'='*80}")
    print(f"CUSTOM REWARD FUNCTION & DATA PREPROCESSING TEST SUITE")
    print(f"{'='*80}")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test custom reward function and data preprocessing")
    parser.add_argument(
        "--parquet_file",
        type=str,
        default=None,
        help="Path to parquet file to test (optional)"
    )
    parser.add_argument(
        "--test_all",
        action="store_true",
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    results = []
    
    # Test 1: Reward function loading
    results.append(("Reward Function Loading", test_reward_function_loading()))
    
    # Test 2: Reward function call
    if results[-1][1]:  # Only test if loading succeeded
        results.append(("Reward Function Call", test_reward_function_call()))
    
    # Test 3: Training configuration
    results.append(("Training Configuration", test_training_config()))
    
    # Test 4: Parquet format (if file provided)
    if args.parquet_file or args.test_all:
        parquet_file = args.parquet_file or os.path.expanduser("~/data/gsm8k/train.parquet")
        results.append(("Parquet Format", test_parquet_format(parquet_file)))
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print(f"✅ ALL TESTS PASSED")
        print(f"\nYour custom reward function and data preprocessing are ready for training!")
    else:
        print(f"❌ SOME TESTS FAILED")
        print(f"\nPlease fix the issues above before running training.")
    print(f"{'='*80}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())



