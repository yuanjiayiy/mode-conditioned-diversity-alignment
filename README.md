<h1 align="center">
  <b>MoDA: Mode-Conditioned Diversity Alignment</b><br>
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.14896v1"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>
  <a href="https://yuanjiayiy.github.io/MODA/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
</p>

# MoDA

**MoDA** (**Mo**de-Conditioned **D**iversity **A**lignment) is a reinforcement learning post-training pipeline that aligns language models while explicitly preserving response diversity, using mode-conditioned rewards on top of a custom [verl](https://github.com/volcengine/verl) reward pipeline.

- **Paper:** [arXiv:2609.14896v1](https://arxiv.org/abs/2609.14896v1)
- **Project website:** [yuanjiayiy.github.io/MODA](https://yuanjiayiy.github.io/MODA/)

---

## Repository Structure

```text
.
├── verl/                    # Core verl RL training library (trainer, workers, models, utils)
├── custom_reward_setup/     # MODA custom reward functions, prompts, and experiment configs
│   ├── configs/             # YAML configs for training runs
│   ├── infinite_chat_reward_function.py
│   ├── skyreward.py
│   ├── prompts.py
│   └── run_qwen3_8b_mix_numbered_role_instruct.sh
├── recipe/                  # Algorithm recipes (DAPO, GSPO, PRIME, etc.)
├── examples/                # Example training scripts for various algorithms
├── docs/                    # Documentation
├── CUSTOM_REWARD_ARCHITECTURE.md   # How the custom reward pipeline is wired into verl
├── QUICK_START_CUSTOM_REWARD.md    # Step-by-step guide to writing/using a custom reward
└── README.md
```

---

## Prerequisites

- A Python environment with this repository's dependencies installed (see `requirements.txt` / `pyproject.toml`); the training entrypoint is `python -m verl.trainer.main_ppo`.
- GPU resources sized for your chosen model (e.g., Qwen3-8B) and rollout backend (vLLM/SGLang/FSDP).
- Optional: set `WANDB_API_KEY` (or place it in `$HOME/.wandb_api_key`) for experiment tracking.

---

## Running MODA Training

Training is launched through the script in `custom_reward_setup/`, which wraps `verl.trainer.main_ppo` with a MODA config.

```bash
CONFIG_NAME=configs/qwen3_8b_mix_numbered_role_instruct_no_thinking \
  bash custom_reward_setup/run_qwen3_8b_mix_numbered_role_instruct.sh
```

- Config files live under `custom_reward_setup/configs/` (e.g. `qwen3_8b_mix_numbered_role_instruct_no_thinking.yaml`, `qwen3_8b_mix_numbered_role_instruct_w_thinking.yaml`).
- `CONFIG_NAME` should be the config path relative to the repo root, **without** the `.yaml` extension.
- If `CONFIG_NAME` is unset, the script falls back to its default config (edit the variable at the top of the script to change it).

For details on how the custom reward function is loaded and invoked during training, see [CUSTOM_REWARD_ARCHITECTURE.md](CUSTOM_REWARD_ARCHITECTURE.md). For a step-by-step guide to writing your own reward function, see [QUICK_START_CUSTOM_REWARD.md](QUICK_START_CUSTOM_REWARD.md).

---

## Acknowledgement

This project builds on [verl](https://github.com/volcengine/verl), an open-source RL training library for LLMs. See [verl's original documentation](https://verl.readthedocs.io/en/latest/) for details on the underlying trainer, workers, and supported algorithms.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{yuan2026shadesbluequalitydiversityalignment,
      title={Forty Shades of Blue: Quality-Diversity Alignment via Mode-Conditioned Reinforcement Learning}, 
      author={Jiayi Yuan and Hangoo Kang and James Jihao Liu and Yejin Choi and Vikram Iyer and Liwei Jiang and Natasha Jaques},
      year={2026},
      eprint={2609.14896},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.14896}, 
}
```
