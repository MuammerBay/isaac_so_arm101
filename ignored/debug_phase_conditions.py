#!/usr/bin/env python3
"""Phase & condition debugging for PicknPlace V1 / V2 environments.

Purpose:
  Quickly inspect whether reward / termination gating conditions ever become true.
  Prints (at an interval) counts and percentages for:
    - cube1_lifted, cube1_at_base
    - cube2_lifted, cube2_stacked
    - stacking_success (termination predicate)
    - phase distribution (using V2 phase detector if available)
  Also reports min/max/mean heights & ee distances to each cube.

Usage examples:
  python debug_phase_conditions.py --version v2 --steps 400 --envs 512
  python debug_phase_conditions.py --version v1 --steps 200 --envs 128

Notes:
  - Does NOT perform learning; uses random (zero) actions so you can observe natural physics / initial feasibility.
  - If nothing ever becomes true (all zeros), your thresholds may be unreachable.
  - For V1 (which may not have dual-cube phases), phase distribution will be skipped unless v2 code is detected.

While training:
  You can run this script in a separate terminal before / during training to validate conditions.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:  # Torch import with helpful fallback
    import torch
except ModuleNotFoundError as e:  # pragma: no cover
    print("[debug_phase_conditions] PyTorch not found in current interpreter.")
    print("Suggestions:")
    print("  1) Run with Isaac Sim python, e.g.: /path/to/isaac-sim/python.sh debug_phase_conditions.py --version v2 --envs 128 --steps 200 --interval 50 --random")
    print("  2) Or install torch into this venv (may be large): pip install 'torch==2.1.0' --extra-index-url https://download.pytorch.org/whl/cu121")
    print("  3) Or deactivate the lightweight project venv (run 'deactivate') so your conda env_isaacsim python (with torch) is used.")
    raise

# Ensure repo source path present
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(REPO_ROOT, 'source')
if SOURCE_PATH not in sys.path:
    sys.path.insert(0, SOURCE_PATH)

os.environ.setdefault("ISAACLAB_HEADLESS", "1")  # allow headless run if display absent

from isaaclab.app import AppLauncher  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

# Import env cfgs lazily (they pull a lot of submodules)
from SO_100.SO_100.tasks.picknplace_v1.picknplace_env_cfg import SoArm100CubePicknPlaceEnvCfg_V1  # noqa: E402
from SO_100.SO_100.tasks.picknplace_v2.picknplace_env_cfg import SoArm100CubePicknPlaceEnvCfg_V2  # noqa: E402

# Try to import v2 phase util
try:
    from SO_100.SO_100.tasks.picknplace_v2.mdp.rewards import get_current_phase, StackingPhase  # type: ignore
except Exception:  # pragma: no cover
    get_current_phase = None  # type: ignore
    StackingPhase = None  # type: ignore


@dataclass
class ConditionStats:
    name: str
    count: int
    pct: float


def tensor_stats(x: torch.Tensor) -> str:
    if x.numel() == 0:
        return 'empty'
    return f"mean={x.mean():.4f} min={x.min():.4f} max={x.max():.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', choices=['v1', 'v2'], default='v2')
    ap.add_argument('--envs', type=int, default=512, help='Number of parallel envs')
    ap.add_argument('--steps', type=int, default=300, help='Simulation steps to run')
    ap.add_argument('--interval', type=int, default=50, help='Print interval (steps)')
    ap.add_argument('--random', action='store_true', help='Use random actions instead of zeros (uniform in [-1,1])')
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args()

    print('=' * 80)
    print(f"PHASE / CONDITION DEBUG | version={args.version} envs={args.envs} steps={args.steps}")
    print('=' * 80)

    # Launch simulator
    app_launcher = AppLauncher()
    sim_app = app_launcher.app

    try:
        if args.version == 'v2':
            env_cfg = SoArm100CubePicknPlaceEnvCfg_V2()
        else:
            env_cfg = SoArm100CubePicknPlaceEnvCfg_V1()
        env_cfg.scene.num_envs = args.envs
        env_cfg.sim.device = args.device

        env: ManagerBasedRLEnv = ManagerBasedRLEnv(cfg=env_cfg)
        obs, _ = env.reset()

        action_shape = env.action_manager.action_spec.shape
        device = env.device
        zero_action = torch.zeros(action_shape, device=device)

        def sample_action():
            if args.random:
                return 2 * torch.rand(action_shape, device=device) - 1
            return zero_action

        print(f"Action shape: {tuple(action_shape)} | Observation policy shape: {obs['policy'].shape}")
        print("Starting stepping...\n")

        for step in range(1, args.steps + 1):
            # Step with zeros (pure hold) - you can randomize if desired
            env.step(sample_action())

            if step % args.interval == 0 or step == args.steps:
                # Collect raw tensors
                cube1 = env.scene['cube1']
                cube2 = env.scene['cube2'] if 'cube2' in env.scene.entities else None
                ee = env.scene['ee_frame']

                cube1_pos = cube1.data.root_pos_w  # [N,3]
                cube2_pos = cube2.data.root_pos_w if cube2 else torch.zeros_like(cube1_pos)
                ee_pos = ee.data.target_pos_w[:, 0, :]

                # Heights / distances
                cube1_height = cube1_pos[:, 2]
                cube2_height = cube2_pos[:, 2]
                c1_ee_dist = torch.norm(cube1_pos - ee_pos, dim=1)
                c2_ee_dist = torch.norm(cube2_pos - ee_pos, dim=1) if cube2 is not None else torch.zeros_like(c1_ee_dist)

                # Thresholds (pull from config if present)
                try:
                    from SO_100.SO_100.tasks.picknplace_v2.config import PICKNPLACE_CFG_V2 as CFG_V2  # type: ignore
                except Exception:  # pragma: no cover
                    CFG_V2 = None  # type: ignore

                minimal_lift_height = 0.04
                placement_tol_xy = 0.05
                stack_align_tol = 0.05
                cube_height = 0.03
                base_pos = torch.tensor([0.25, -0.15, 0.0575], device=device)  # fallback typical
                if CFG_V2 is not None:
                    minimal_lift_height = getattr(CFG_V2, 'MINIMAL_LIFT_HEIGHT', minimal_lift_height)
                    placement_tol_xy = getattr(CFG_V2, 'PLACEMENT_TOLERANCE_XY', placement_tol_xy)
                    stack_align_tol = getattr(CFG_V2, 'STACK_ALIGNMENT_TOLERANCE', stack_align_tol)
                    base_pos = torch.tensor(getattr(CFG_V2, 'STACK_BASE_POS', base_pos.tolist()), device=device)
                    cube_height = getattr(CFG_V2, 'CUBE1_SIZE', (0.03, 0.03, 0.03))[2]

                cube1_lifted = cube1_height > minimal_lift_height
                cube1_at_base = (
                    (torch.abs(cube1_pos[:, 0] - base_pos[0]) <= placement_tol_xy)
                    & (torch.abs(cube1_pos[:, 1] - base_pos[1]) <= placement_tol_xy)
                    & (torch.abs(cube1_pos[:, 2] - base_pos[2]) <= 0.01)
                )

                cube2_lifted = cube2_height > minimal_lift_height if cube2 is not None else torch.zeros_like(cube1_lifted)
                expected_cube2_z = cube1_pos[:, 2] + cube_height
                cube2_stacked = torch.zeros_like(cube1_lifted)
                if cube2 is not None:
                    cube2_stacked = (
                        (torch.abs(cube2_pos[:, 0] - cube1_pos[:, 0]) <= stack_align_tol)
                        & (torch.abs(cube2_pos[:, 1] - cube1_pos[:, 1]) <= stack_align_tol)
                        & (torch.abs(cube2_pos[:, 2] - expected_cube2_z) <= 0.01)
                    )

                stacking_success = cube1_at_base & cube2_stacked

                # Phase distribution (v2 only if available)
                phase_counts = {}
                if args.version == 'v2' and get_current_phase is not None:
                    phases = get_current_phase(env)
                    for p_val in phases.unique():
                        p_int = int(p_val.item())
                        phase_counts[p_int] = int((phases == p_val).sum().item())

                def summarize(mask: torch.Tensor, name: str) -> ConditionStats:
                    cnt = int(mask.sum().item())
                    pct = (cnt / env.num_envs) * 100.0
                    return ConditionStats(name, cnt, pct)

                stats = [
                    summarize(cube1_lifted, 'cube1_lifted'),
                    summarize(cube1_at_base, 'cube1_at_base'),
                    summarize(cube2_lifted, 'cube2_lifted'),
                    summarize(cube2_stacked, 'cube2_stacked'),
                    summarize(stacking_success, 'stacking_success'),
                ]

                print(f"Step {step:5d} | Conditions:" + ' '.join([f"{s.name}={s.count}({s.pct:5.1f}%)" for s in stats]))
                print(f"   cube1_height: {tensor_stats(cube1_height)} | cube2_height: {tensor_stats(cube2_height)}")
                print(f"   c1_ee_dist: {tensor_stats(c1_ee_dist)} | c2_ee_dist: {tensor_stats(c2_ee_dist)}")
                if phase_counts:
                    phase_line = ' '.join([
                        f"P{p}={c}({c / env.num_envs * 100:4.1f}%)" for p, c in sorted(phase_counts.items())
                    ])
                    print(f"   phases: {phase_line}")
                print()

        print('Finished stepping. Summary above.')

    except Exception as e:  # pragma: no cover
        print(f"ERROR during debug: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        try:
            sim_app.close()
        except Exception:
            pass


if __name__ == '__main__':  # pragma: no cover
    main()
