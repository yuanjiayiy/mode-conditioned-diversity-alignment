#!/usr/bin/env python3
"""Quick test for G-Vendi implementation."""

import sys

# Add parent for imports
sys.path.insert(0, "/mmfs1/gscratch/socialrl/y9/verl_collaboration")

from custom_reward_setup.g_vendi import (
    compute_g_vendi_score,
    compute_g_vendi_score_from_gradients,
    _vendi_score_from_projected,
)
import torch


def test_vendi_from_projected():
    """Test Vendi score from synthetic projected gradients."""
    # Identical vectors -> low diversity (entropy ~ 0)
    G_same = torch.randn(1, 1024).repeat(10, 1)
    score_same = _vendi_score_from_projected(G_same)
    print(f"Identical vectors G-Vendi: {score_same:.2f} (expected ~1)")

    # Orthogonal/diverse vectors -> high diversity
    G_diverse = torch.randn(50, 1024)
    G_diverse = G_diverse / G_diverse.norm(dim=1, keepdim=True)
    score_diverse = _vendi_score_from_projected(G_diverse)
    print(f"Diverse random vectors G-Vendi: {score_diverse:.2f} (expected > 1)")

    assert score_diverse > score_same, "Diverse should score higher"
    print("test_vendi_from_projected: OK")


def test_compute_g_vendi_score():
    """Test full G-Vendi with real model (requires GPU and model download)."""
    prompts = [
        "What is 2 + 2?",
        "Explain photosynthesis briefly.",
        "Write a haiku about coding.",
    ]
    responses = [
        "2 + 2 equals 4.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "Code flows like rivers, bugs hide in the shadows, debug until dawn.",
    ]

    print("Computing G-Vendi (this may take a minute on first run)...")
    score = compute_g_vendi_score(
        prompts=prompts,
        responses=responses,
        proxy_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        proj_dim=1024,
        max_length=512,
    )
    print(f"G-Vendi score: {score:.2f}")
    assert score >= 0, "Score should be non-negative"
    print("test_compute_g_vendi_score: OK")


if __name__ == "__main__":
    test_vendi_from_projected()
    print()
    test_compute_g_vendi_score()
