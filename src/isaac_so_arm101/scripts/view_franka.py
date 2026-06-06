# Copyright (c) 2024-2025, Muammer Bay (LycheeAI)
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Franka viewer for Isaac Lab.

This script intentionally stays separate from the SO-ARM task files so the Franka
arm can be launched in the simulator without modifying the SO101 task setup.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch the built-in Franka reach task in Isaac Lab.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of Franka arms to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# Import the Franka task package so its gym environments are registered before lookup.
import isaaclab_tasks.manager_based.manipulation.reach.config.franka  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


TASK_NAME = "Isaac-Reach-Franka-Play-v0"


def main():
    """Launch the Franka arm in the simulator and keep it alive with zero actions."""
    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(TASK_NAME, cfg=env_cfg)

    print(f"[INFO]: Task: {TASK_NAME}")
    print(f"[INFO]: Observation space: {env.observation_space}")
    print(f"[INFO]: Action space: {env.action_space}")

    env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
