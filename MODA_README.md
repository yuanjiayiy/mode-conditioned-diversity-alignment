MODA README

This document explains how to run the training script in the `custom_reward_setup` folder and how to specify the configuration file (located in the `configs` directory).

**What this script runs**: [custom_reward_setup/run_qwen3_8b_mix_numbered_role_instruct.sh](custom_reward_setup/run_qwen3_8b_mix_numbered_role_instruct.sh) launches `verl.trainer.main_ppo` with a config specified by the `CONFIG_NAME` variable.

Prerequisites
- **Python environment** with repository dependencies installed (use your usual venv/conda). The script calls `python -m verl.trainer.main_ppo`.
- Optional: set `WANDB_API_KEY` or put your key in `$HOME/.wandb_api_key` if you want W&B logging.

Specifying the config
- Place your YAML config files under the repository `configs` directory (e.g. `configs/my_experiment.yaml`).
- The script determines the config by the `CONFIG_NAME` variable. The script appends `.yaml` automatically, so provide the path without the extension.

Examples
- Run using a config file `configs/my_experiment.yaml` from the repository root:

```bash
CONFIG_NAME=configs/my_experiment bash custom_reward_setup/run_qwen3_8b_mix_numbered_role_instruct.sh
```

- Alternatively, edit the `CONFIG_NAME` value at the top of [custom_reward_setup/run_qwen3_8b_mix_numbered_role_instruct.sh](custom_reward_setup/run_qwen3_8b_mix_numbered_role_instruct.sh) to set a different default config (note: value should omit the `.yaml` extension).

Notes
- The script sets a default `CONFIG_NAME` of `customqwen3_8b_mix_numbered_role_instruct_no_thinking`. If you don't override it, the script will look for `customqwen3_8b_mix_numbered_role_instruct_no_thinking.yaml` in the working directory unless you include a directory component (e.g. `configs/customqwen3_8b_mix_numbered_role_instruct_no_thinking`).
- If you want the config directory to be `configs`, use `CONFIG_NAME=configs/<your_config_base>` when running.

If you want, I can also add a small wrapper script to pass a `--config` argument or validate that the referenced config exists before launching.
