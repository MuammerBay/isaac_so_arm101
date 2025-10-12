# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

# Import central configuration
from ..config import PICKNPLACE_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return distance < threshold


def object_placement_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for successful object placement on red target area using central config.
    
    Returns boolean tensor indicating which environments have successfully placed the object on target area.
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object position
    obj_pos = object.data.root_pos_w
    success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    for i in range(env.num_envs):
        obj_position = obj_pos[i].cpu().numpy().tolist()
        is_on_target = PICKNPLACE_CFG.is_object_in_target_area(obj_position)
        success[i] = is_on_target
    
    return success


def object_dropping(
    env: ManagerBasedRLEnv,
    drop_height_threshold: float = 0.005,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for object dropping below threshold."""
    object: RigidObject = env.scene[object_cfg.name]
    
    is_dropped = object.data.root_pos_w[:, 2] < drop_height_threshold
    
    return is_dropped
