# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

# Import central configuration
from ..config import PICKNPLACE_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Combine rewards multiplicatively
    return reach_reward * lift_reward


# ===============================================================================
# Pick & Place Specific Rewards
# ===============================================================================

def object_target_distance(
    env: ManagerBasedRLEnv,
    std: float,
    target_position: list[float] | torch.Tensor,
    minimal_height: float = 0.04,  # Require lifting
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for moving the object close to a target position - ONLY if lifted."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # Convert target position to tensor if it's a list
    if isinstance(target_position, list):
        target_pos = torch.tensor(target_position, device=env.device, dtype=torch.float32)
        target_pos = target_pos.unsqueeze(0).expand(env.num_envs, -1)
    else:
        target_pos = target_position
    
    # Distance from object to target position: (num_envs,)
    distance = torch.norm(object.data.root_pos_w[:, :3] - target_pos, dim=1)
    
    # CRITICAL: Only reward transport if object is lifted above minimal height
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    transport_reward = 1 - torch.tanh(distance / std)
    
    # Zero reward if not lifted (prevents dragging)
    return is_lifted.float() * transport_reward


def object_placement_reward(
    env: ManagerBasedRLEnv,
    target_position: list[float] | torch.Tensor,
    placement_tolerance: float = 0.05,
    height_tolerance: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for precise object placement at target location."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # Convert target position to tensor if it's a list
    if isinstance(target_position, list):
        target_pos = torch.tensor(target_position, device=env.device, dtype=torch.float32)
        target_pos = target_pos.unsqueeze(0).expand(env.num_envs, -1)
    else:
        target_pos = target_position
    
    # Calculate distances
    obj_pos = object.data.root_pos_w[:, :3]
    xy_distance = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=1)
    z_distance = torch.abs(obj_pos[:, 2] - target_pos[:, 2])
    
    # Check if object is within placement tolerance
    xy_in_tolerance = xy_distance < placement_tolerance
    z_in_tolerance = z_distance < height_tolerance
    
    # Combined placement success
    placement_success = xy_in_tolerance & z_in_tolerance
    
    return placement_success.float()


def object_drop_penalty(
    env: ManagerBasedRLEnv,
    drop_height_threshold: float = 0.005,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for dropping the object (object falling below threshold)."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # Check if object has dropped below threshold
    is_dropped = object.data.root_pos_w[:, 2] < drop_height_threshold
    
    return -is_dropped.float()


def gripper_object_alignment(
    env: ManagerBasedRLEnv,
    std: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for proper gripper-object alignment before grasping."""
    from isaaclab.sensors import FrameTransformer
    
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Get positions
    obj_pos = object.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    
    # Calculate alignment (focus on XY plane for approach)
    xy_distance = torch.norm(obj_pos[:, :2] - ee_pos[:, :2], dim=1)
    
    return 1 - torch.tanh(xy_distance / std)


def object_dragging_penalty(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    velocity_threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for dragging object on ground instead of lifting."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # Check if object is on ground
    is_on_ground = object.data.root_pos_w[:, 2] < minimal_height
    
    # Check if object is moving (being dragged)
    velocity_magnitude = torch.norm(object.data.root_lin_vel_w[:, :2], dim=1)  # XY velocity
    is_moving = velocity_magnitude > velocity_threshold
    
    # Penalty if object is being dragged on ground
    is_dragging = is_on_ground & is_moving
    
    return -is_dragging.float()


def object_on_target_area_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for successfully placing object on the red target area using central config."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object position
    obj_pos = object.data.root_pos_w
    
    # Check if object is on target area for each environment
    rewards = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(env.num_envs):
        obj_position = obj_pos[i].cpu().numpy().tolist()
        is_on_target = PICKNPLACE_CFG.is_object_in_target_area(obj_position)
        rewards[i] = 1.0 if is_on_target else 0.0
    
    return rewards


def distance_to_target_area_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    minimal_height: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward based on distance to target area center, only when lifted."""
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object position
    obj_pos = object.data.root_pos_w
    
    # Target area position from central config
    target_pos = torch.tensor(
        PICKNPLACE_CFG.get_target_area_position(), 
        device=env.device, 
        dtype=torch.float32
    ).unsqueeze(0).expand(env.num_envs, -1)
    
    # Calculate distance
    distance = torch.norm(obj_pos - target_pos, dim=1)
    
    # Only reward if object is lifted
    is_lifted = obj_pos[:, 2] > minimal_height
    distance_reward = 1.0 - torch.tanh(distance / std)
    
    return is_lifted.float() * distance_reward


def gripper_release_after_placement(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for releasing gripper after successful placement.
    
    This reward encourages the robot to move the gripper away from the object
    after it has been successfully placed on the target area.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Get object position
    obj_pos = object.data.root_pos_w
    
    # Check if object is on target area for each environment
    is_on_target = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(env.num_envs):
        obj_position = obj_pos[i].cpu().numpy().tolist()
        is_on_target[i] = PICKNPLACE_CFG.is_object_in_target_area(obj_position)
    
    # Get gripper (end-effector) position
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    
    # Calculate distance between gripper and object
    distance = torch.norm(obj_pos[:, :3] - ee_pos, dim=1)
    
    # Reward moving gripper away from object when object is on target
    # Use tanh to normalize the reward between 0 and 1
    release_reward = is_on_target * torch.tanh(distance / 0.1)
    
    return release_reward
