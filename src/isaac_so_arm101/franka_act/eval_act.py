# Copyright (c) 2024-2025, Muammer Bay (LycheeAI)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained ACT checkpoint on Franka in Isaac Lab.")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory created by train_act.py.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0", help="Isaac task to run.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many environment steps. 0 keeps running.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from isaac_so_arm101.franka_act.common import flatten_policy_obs_dict, load_metadata
from isaac_so_arm101.franka_act.runtime import apply_streaming_env_workarounds, ensure_isaac_asset_root

ensure_isaac_asset_root()

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE


def main() -> None:
    metadata = load_metadata(args_cli.checkpoint_dir)
    state_keys = metadata["state_keys"]

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.observations.policy.concatenate_terms = False
    if getattr(args_cli, "livestream", 0):
        apply_streaming_env_workarounds(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    policy = ACTPolicy.from_pretrained(args_cli.checkpoint_dir)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args_cli.checkpoint_dir)
    policy.reset()
    policy.eval()

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            env_state = flatten_policy_obs_dict(obs["policy"], state_keys)
            env_state_t = torch.from_numpy(env_state)
            batch = {
                OBS_STATE: env_state_t,
                OBS_ENV_STATE: env_state_t,
            }
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            action = action.to(env.unwrapped.device)

            obs, reward, terminated, truncated, info = env.step(action)

            if step % 50 == 0:
                print(f"[EVAL] step={step:05d} reward={float(reward[0]):.4f}")

            if terminated.any() or truncated.any():
                print(
                    f"[EVAL] episode reset at step={step} "
                    f"terminated={bool(terminated.any())} truncated={bool(truncated.any())}"
                )
                policy.reset()

            step += 1
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
