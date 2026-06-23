# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate Lift (pick-to-goal) policy success rate with matplotlib + HTML report.

Success (per episode): object lifted (world z > min_height) AND planar goal distance < pos_threshold.
Terminal state is sampled by wrapping :meth:`InteractiveScene.reset` right before physics buffers reset.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import types
from dataclasses import dataclass, field

import torch
from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate Lift-Cube policy success rate.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0", help="Play task id.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs for rollouts.")
parser.add_argument(
    "--pos_threshold",
    type=float,
    default=0.05,
    help="Max object-to-goal position error (m) for success.",
)
parser.add_argument(
    "--min_height",
    type=float,
    default=0.025,
    help="Minimum object world z (m) to count as lifted (matches training reward).",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=512,
    help="Stop after this many completed episodes.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/eval_lift",
    help="Directory for success_rate.png and report.html",
)
parser.add_argument("--seed", type=int, default=0, help="Env seed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import platform
from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(f"Need rsl-rl-lib>={RSL_RL_VERSION}, found {installed_version}")
    sys.exit(1)

import gymnasium as gym
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401


def _goal_distance_and_lifted(env, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )
    dist = torch.norm(des_pos_w - obj.data.root_pos_w[:, :3], dim=1)
    lifted = obj.data.root_pos_w[:, 2] > args_cli.min_height
    return dist, lifted


@dataclass
class EpisodeStats:
    successes: list[bool] = field(default_factory=list)
    dists: list[float] = field(default_factory=list)
    lifted: list[bool] = field(default_factory=list)
    term_timeout: list[bool] = field(default_factory=list)

    def record(
        self,
        env_ids: torch.Tensor,
        dist: torch.Tensor,
        lifted: torch.Tensor,
        reset_time_outs: torch.Tensor | None,
    ):
        ids = env_ids.flatten().tolist()
        for i in ids:
            d = float(dist[i].item())
            l = bool(lifted[i].item())
            self.dists.append(d)
            self.lifted.append(l)
            ok = l and (d < args_cli.pos_threshold)
            self.successes.append(ok)
            if reset_time_outs is not None:
                self.term_timeout.append(bool(reset_time_outs[i].item()))


def main_inner(env, vec_env: RslRlVecEnvWrapper, policy, stats: EpisodeStats, menv):
    orig_scene_reset = menv.scene.reset

    def wrapped_reset(self, env_ids, *args, **kwargs):
        if env_ids is not None and len(env_ids) > 0:
            eids = env_ids
            if not isinstance(eids, torch.Tensor):
                eids = torch.as_tensor(eids, device=menv.device, dtype=torch.long)
            lens = menv.episode_length_buf[eids]
            valid = eids[lens > 0]
            if valid.numel() > 0:
                dist, lifted = _goal_distance_and_lifted(menv, "object_pose")
                stats.record(valid, dist, lifted, menv.reset_time_outs)
        return orig_scene_reset(env_ids, *args, **kwargs)

    menv.scene.reset = types.MethodType(wrapped_reset, menv.scene)

    obs = vec_env.get_observations()
    episodes_done = 0
    max_steps = 10_000_000
    for _step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _rew, dones, _extras = vec_env.step(actions)
        episodes_done = len(stats.successes)
        if episodes_done >= args_cli.max_episodes:
            break
        if _step > 0 and _step % 5000 == 0:
            print(f"[eval] steps={_step}, episodes={episodes_done}")

    menv.scene.reset = orig_scene_reset


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        from isaaclab.utils.assets import retrieve_file_path

        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    menv = vec_env.unwrapped

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=vec_env.device)

    stats = EpisodeStats()
    try:
        main_inner(env, vec_env, policy, stats, menv)
    finally:
        env.close()

    n = len(stats.successes)
    if n == 0:
        print("[ERROR] No episodes recorded (scene.reset hook did not fire?).")
        return

    succ = sum(stats.successes)
    rate = 100.0 * succ / n
    print(f"[RESULT] episodes={n}, success={succ}, rate={rate:.2f}%")
    print(f"         pos_threshold={args_cli.pos_threshold} m, min_height={args_cli.min_height} m")

    out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "success_rate.png")
    html_path = os.path.join(out_dir, "report.html")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(["fail", "success"], [n - succ, succ], color=["#c44e52", "#4c72b0"])
        axes[0].set_ylabel("Episodes")
        axes[0].set_title(f"Lift goal success ({rate:.1f}%)")

        window = min(64, max(8, n // 8))
        roll = []
        for i in range(n):
            a = max(0, i - window + 1)
            chunk = stats.successes[a : i + 1]
            roll.append(100.0 * sum(chunk) / len(chunk))
        axes[1].plot(roll, color="#55a868")
        axes[1].set_ylim(0, 100)
        axes[1].set_xlabel("Episode index")
        axes[1].set_ylabel(f"Rolling success % (w={window})")
        axes[1].set_title("Rolling success rate")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()
    except Exception as e:
        print(f"[WARN] matplotlib failed ({e}); skip PNG.")
        png_path = ""

    buf = io.BytesIO()
    if png_path and os.path.isfile(png_path):
        with open(png_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("ascii")
    else:
        b64 = ""

    dist_mean = sum(stats.dists) / n
    lifted_rate = 100.0 * sum(stats.lifted) / n
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Lift eval</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 960px; }}
table {{ border-collapse: collapse; margin-top: 1rem; }}
td, th {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
</style></head><body>
<h1>Lift-Cube evaluation</h1>
<p><b>Checkpoint:</b> {resume_path}</p>
<p><b>Episodes:</b> {n} &nbsp;|&nbsp; <b>Success rate:</b> {rate:.2f}%</p>
<p><b>Criteria:</b> object z &gt; {args_cli.min_height} m and ||p_obj − p_goal|| &lt; {args_cli.pos_threshold} m at episode end.</p>
<p><b>Lifted-only rate:</b> {lifted_rate:.2f}% &nbsp;|&nbsp; <b>Mean terminal ||error||:</b> {dist_mean:.4f} m</p>
{f'<img alt="plots" style="max-width:100%" src="data:image/png;base64,{b64}"/>' if b64 else '<p>(No plot image.)</p>'}
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Successes</td><td>{succ}</td></tr>
<tr><td>Failures</td><td>{n - succ}</td></tr>
<tr><td>Mean goal distance (terminal)</td><td>{dist_mean:.4f} m</td></tr>
</table>
</body></html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] Wrote {html_path}")
    if png_path and os.path.isfile(png_path):
        print(f"[INFO] Wrote {png_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
