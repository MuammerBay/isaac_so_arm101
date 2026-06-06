# Copyright (c) 2024-2025, Muammer Bay (LycheeAI)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from isaac_so_arm101.franka_act.common import (
    IsaacActDataset,
    build_lightweight_act_config,
    compute_dataset_stats,
    load_isaac_hdf5_episodes,
    save_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight ACT policy on Isaac HDF5 demos.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the Isaac HDF5 demo file.")
    parser.add_argument("--output_dir", type=str, default="./outputs/franka_act", help="Directory for checkpoints.")
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0", help="Task name metadata.")
    parser.add_argument("--action_key", type=str, default=None, help="Explicit action dataset key to use.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--steps", type=int, default=3000, help="Number of optimizer steps.")
    parser.add_argument("--log_freq", type=int, default=50, help="Logging frequency in steps.")
    parser.add_argument("--chunk_size", type=int, default=16, help="Number of future actions to supervise.")
    parser.add_argument("--n_action_steps", type=int, default=8, help="Number of actions executed per query.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes, state_keys, action_key = load_isaac_hdf5_episodes(args.dataset_file, action_key=args.action_key)
    dataset = IsaacActDataset(episodes, chunk_size=args.chunk_size)
    stats = compute_dataset_stats(episodes)

    state_dim = episodes[0].env_state.shape[-1]
    action_dim = episodes[0].actions.shape[-1]

    cfg = build_lightweight_act_config(
        state_dim=state_dim,
        action_dim=action_dim,
        device=args.device,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        learning_rate=args.learning_rate,
    )
    policy = ACTPolicy(cfg).to(cfg.device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)
    policy.train()

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.learning_rate, weight_decay=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=cfg.device != "cpu",
    )

    print(
        f"[INFO] Loaded {len(episodes)} episodes, {len(dataset)} training frames, "
        f"state_dim={state_dim}, action_dim={action_dim}, action_key={action_key}"
    )

    step = 0
    while step < args.steps:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_freq == 0:
                print(
                    f"[TRAIN] step={step:05d} loss={loss.item():.4f} "
                    f"l1={loss_dict['l1_loss']:.4f}"
                )

            step += 1
            if step >= args.steps:
                break

    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    save_metadata(
        output_dir,
        dataset_file=args.dataset_file,
        task=args.task,
        state_keys=state_keys,
        action_key=action_key,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
    )
    print(f"[INFO] Saved ACT checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
