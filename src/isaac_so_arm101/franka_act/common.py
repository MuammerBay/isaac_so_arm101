# Copyright (c) 2024-2025, Muammer Bay (LycheeAI)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _sorted_hdf5_leaves(group: h5py.Group, prefix: str = "") -> list[tuple[str, h5py.Dataset]]:
    leaves: list[tuple[str, h5py.Dataset]] = []
    for key in sorted(group.keys()):
        value = group[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Group):
            leaves.extend(_sorted_hdf5_leaves(value, path))
        else:
            leaves.append((path, value))
    return leaves


def flatten_policy_obs_group(obs_group: h5py.Group, state_keys: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    leaves = _sorted_hdf5_leaves(obs_group)
    if not leaves:
        raise ValueError("No observation leaves found under the HDF5 obs group.")

    if state_keys is None:
        state_keys = [name for name, _ in leaves]

    dataset_map = {name: np.asarray(dataset, dtype=np.float32) for name, dataset in leaves}
    missing_keys = [name for name in state_keys if name not in dataset_map]
    if missing_keys:
        raise KeyError(f"Missing observation keys in dataset: {missing_keys}")

    features = [dataset_map[name].reshape(dataset_map[name].shape[0], -1) for name in state_keys]
    env_state = np.concatenate(features, axis=-1).astype(np.float32)
    return env_state, state_keys


def flatten_policy_obs_dict(obs_dict: dict, state_keys: list[str]) -> np.ndarray:
    flattened: dict[str, np.ndarray] = {}

    def recurse(node: dict | torch.Tensor, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key in sorted(node.keys()):
                child_prefix = f"{prefix}/{key}" if prefix else key
                recurse(node[key], child_prefix)
            return

        value = node
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy().astype(np.float32)
        else:
            array = np.asarray(value, dtype=np.float32)
        flattened[prefix] = array.reshape(array.shape[0], -1)

    recurse(obs_dict)

    missing_keys = [name for name in state_keys if name not in flattened]
    if missing_keys:
        raise KeyError(f"Missing observation keys at inference time: {missing_keys}")

    return np.concatenate([flattened[name] for name in state_keys], axis=-1).astype(np.float32)


def resolve_action_key(episode_group: h5py.Group, action_key: str | None) -> str:
    if action_key is not None:
        if action_key not in episode_group:
            raise KeyError(f"Action key '{action_key}' not found in episode.")
        return action_key
    for candidate in ("actions", "processed_actions"):
        if candidate in episode_group:
            return candidate
    raise KeyError("Could not find an action dataset. Expected one of: actions, processed_actions.")


@dataclass
class IsaacEpisode:
    env_state: np.ndarray
    actions: np.ndarray


def load_isaac_hdf5_episodes(
    dataset_path: str | Path,
    *,
    action_key: str | None = None,
    state_keys: list[str] | None = None,
    min_steps: int = 2,
) -> tuple[list[IsaacEpisode], list[str], str]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    episodes: list[IsaacEpisode] = []
    resolved_state_keys = state_keys
    resolved_action_key = action_key

    with h5py.File(dataset_path, "r") as handle:
        data_group = handle["data"]
        for episode_name in sorted(data_group.keys()):
            episode_group = data_group[episode_name]
            if "obs" not in episode_group:
                continue
            resolved_action_key = resolve_action_key(episode_group, resolved_action_key)
            env_state, resolved_state_keys = flatten_policy_obs_group(episode_group["obs"], resolved_state_keys)
            actions = np.asarray(episode_group[resolved_action_key], dtype=np.float32)
            num_steps = min(len(env_state), len(actions))
            if num_steps < min_steps:
                continue
            episodes.append(IsaacEpisode(env_state=env_state[:num_steps], actions=actions[:num_steps]))

    if not episodes:
        raise ValueError(f"No usable episodes found in dataset: {dataset_path}")
    if resolved_state_keys is None or resolved_action_key is None:
        raise RuntimeError("Failed to resolve state/action keys from dataset.")

    return episodes, resolved_state_keys, resolved_action_key


class IsaacActDataset(Dataset):
    def __init__(self, episodes: list[IsaacEpisode], chunk_size: int):
        self.episodes = episodes
        self.chunk_size = chunk_size
        self.index: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(episodes):
            for step_idx in range(len(episode.actions)):
                self.index.append((episode_idx, step_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_idx, step_idx = self.index[idx]
        episode = self.episodes[episode_idx]
        action_dim = episode.actions.shape[-1]

        env_state = torch.from_numpy(episode.env_state[step_idx]).float()

        action_chunk = torch.zeros((self.chunk_size, action_dim), dtype=torch.float32)
        action_is_pad = torch.ones((self.chunk_size,), dtype=torch.bool)

        horizon_end = min(step_idx + self.chunk_size, len(episode.actions))
        valid_actions = torch.from_numpy(episode.actions[step_idx:horizon_end]).float()
        valid_len = valid_actions.shape[0]
        action_chunk[:valid_len] = valid_actions
        action_is_pad[:valid_len] = False

        return {
            OBS_STATE: env_state,
            OBS_ENV_STATE: env_state,
            ACTION: action_chunk,
            "action_is_pad": action_is_pad,
            "episode_index": torch.tensor(episode_idx, dtype=torch.int64),
            "index": torch.tensor(step_idx, dtype=torch.int64),
        }


def compute_dataset_stats(episodes: list[IsaacEpisode]) -> dict[str, dict[str, torch.Tensor]]:
    env_states = np.concatenate([episode.env_state for episode in episodes], axis=0)
    actions = np.concatenate([episode.actions for episode in episodes], axis=0)

    def make_stats(array: np.ndarray) -> dict[str, torch.Tensor]:
        mean = torch.from_numpy(array.mean(axis=0).astype(np.float32))
        std = torch.from_numpy(array.std(axis=0).astype(np.float32))
        std = torch.clamp(std, min=1e-6)
        return {"mean": mean, "std": std}

    return {
        OBS_STATE: make_stats(env_states),
        OBS_ENV_STATE: make_stats(env_states),
        ACTION: make_stats(actions),
    }


def build_lightweight_act_config(
    *,
    state_dim: int,
    action_dim: int,
    device: str,
    chunk_size: int,
    n_action_steps: int,
    learning_rate: float,
) -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(state_dim,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        device=device,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        use_vae=False,
        dim_model=256,
        n_heads=4,
        dim_feedforward=1024,
        n_encoder_layers=3,
        n_decoder_layers=1,
        dropout=0.1,
        optimizer_lr=learning_rate,
        optimizer_lr_backbone=learning_rate,
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ENV: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
    )


def save_metadata(
    output_dir: str | Path,
    *,
    dataset_file: str,
    task: str,
    state_keys: list[str],
    action_key: str,
    chunk_size: int,
    n_action_steps: int,
) -> None:
    output_dir = Path(output_dir)
    metadata = {
        "dataset_file": str(dataset_file),
        "task": task,
        "state_keys": state_keys,
        "action_key": action_key,
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
    }
    with open(output_dir / "franka_act_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_metadata(checkpoint_dir: str | Path) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "franka_act_metadata.json", "r", encoding="utf-8") as file:
        return json.load(file)
