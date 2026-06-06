# Copyright (c) 2024-2025, Muammer Bay (LycheeAI)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import os
import sys


DEFAULT_ISAAC_ASSET_ROOT = os.environ.get(
    "ISAAC_ASSET_ROOT",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
)
_STREAMING_VIEWPORT_PATCHED = False


def ensure_isaac_asset_root() -> str:
    """Populate Isaac Sim asset-root settings when the streaming app leaves them unset."""
    import carb

    settings = carb.settings.get_settings()
    asset_root = settings.get("/persistent/isaac/asset_root/cloud")
    if asset_root:
        _patch_imported_asset_modules(asset_root)
        return asset_root

    asset_root = DEFAULT_ISAAC_ASSET_ROOT
    settings.set("/persistent/isaac/asset_root/default", asset_root)
    settings.set("/persistent/isaac/asset_root/cloud", asset_root)
    settings.set("/persistent/isaac/asset_root/nvidia", asset_root)
    _patch_imported_asset_modules(asset_root)
    print(f"[INFO] Set Isaac asset root: {asset_root}")
    return asset_root


def apply_streaming_env_workarounds(env_cfg) -> None:
    """Apply Isaac Lab workarounds for livestream runs that can hang during viewport tracking setup."""
    _disable_viewport_camera_controller_for_streaming()
    env_cfg.ui_window_class_type = None
    env_cfg.wait_for_textures = False
    if getattr(env_cfg, "viewer", None) is not None:
        env_cfg.viewer.origin_type = "world"
    print("[INFO] Applied streaming env workarounds: disabled viewport tracking and Isaac Lab UI window.")


def _patch_imported_asset_modules(asset_root: str) -> None:
    """Refresh already-imported Isaac Lab asset constants that were initialized before the asset root existed."""
    isaac_nucleus_dir = f"{asset_root}/Isaac"
    isaaclab_nucleus_dir = f"{isaac_nucleus_dir}/IsaacLab"

    assets_module = sys.modules.get("isaaclab.utils.assets")
    if assets_module is not None:
        assets_module.NUCLEUS_ASSET_ROOT_DIR = asset_root
        assets_module.NVIDIA_NUCLEUS_DIR = f"{asset_root}/NVIDIA"
        assets_module.ISAAC_NUCLEUS_DIR = isaac_nucleus_dir
        assets_module.ISAACLAB_NUCLEUS_DIR = isaaclab_nucleus_dir

    franka_module = sys.modules.get("isaaclab_assets.robots.franka")
    if franka_module is not None:
        franka_module.FRANKA_PANDA_CFG.spawn.usd_path = (
            f"{isaaclab_nucleus_dir}/Robots/FrankaEmika/panda_instanceable.usd"
        )
        franka_module.FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path = (
            f"{isaaclab_nucleus_dir}/Robots/FrankaEmika/panda_instanceable.usd"
        )
        franka_module.FRANKA_ROBOTIQ_GRIPPER_CFG.spawn.usd_path = (
            f"{isaac_nucleus_dir}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        )


def _disable_viewport_camera_controller_for_streaming() -> None:
    global _STREAMING_VIEWPORT_PATCHED
    if _STREAMING_VIEWPORT_PATCHED:
        return

    class _NoOpViewportCameraController:
        def __init__(self, *args, **kwargs):
            self.cfg = args[1] if len(args) > 1 else kwargs.get("cfg")

        def set_view_env_index(self, *args, **kwargs):
            return None

        def update_view_to_world(self, *args, **kwargs):
            return None

        def update_view_to_env(self, *args, **kwargs):
            return None

        def update_view_to_asset_root(self, *args, **kwargs):
            return None

        def update_view_to_asset_body(self, *args, **kwargs):
            return None

        def update_view_location(self, *args, **kwargs):
            return None

    for module_name in (
        "isaaclab.envs.ui",
        "isaaclab.envs.manager_based_env",
        "isaaclab.envs.direct_rl_env",
        "isaaclab.envs.direct_marl_env",
    ):
        module = importlib.import_module(module_name)
        module.ViewportCameraController = _NoOpViewportCameraController

    _STREAMING_VIEWPORT_PATCHED = True
    print("[INFO] Disabled Isaac Lab viewport camera controller for streaming.")
