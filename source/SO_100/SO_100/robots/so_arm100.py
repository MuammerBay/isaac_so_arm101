# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the SO-ARM100 5-DOF robot arm for the livestream.

The following configuration is available:

* :obj:`SO_ARM100_CFG`: SO-ARM100 robot arm configuration.
        ->  converted from the xacro of this repository:
        https://github.com/JafarAbdi/ros2_so_arm100
"""

import os
import math

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

##
# Configuration
##

SO_ARM100_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Robots/so_arm100/so_arm100/so_arm100.usd",
        activate_contact_sensors=False,                 # Adjust based on need
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(0.7071068, 0.0, 0.0, 0.7071068),   # Quaternion for 90 degrees rotation around Y-axis
        joint_pos={
            "shoulder_rotation":    0.1,
            "shoulder_pitch":       0.5,
            "elbow":                0.0,
            "wrist_pitch":          0.0,
            "wrist_roll":           0.0,
            "gripper":              0.3,        # Middle position to make movement more apparent
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Shoulder Rotation moves: ALL masses                   (~0.8kg total)
        # Shoulder Pitch    moves: Everything except base       (~0.65kg)
        # Elbow             moves: Lower arm, wrist, gripper    (~0.38kg)
        # Wrist Pitch       moves: Wrist and gripper            (~0.24kg)
        # Wrist Roll        moves: Gripper assembly             (~0.14kg)
        # Gripper           moves: Only moving jaw              (~0.034kg)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow", "wrist_.*"],
            effort_limit_sim = 1.9,
            velocity_limit_sim = 1.5,
            stiffness={
                "shoulder_rotation":    200.0,  # Highest - moves all mass
                "shoulder_pitch":       170.0,  # Slightly less than rotation
                "elbow":                120.0,  # Reduced based on less mass
                "wrist_pitch":          80.0,   # Reduced for less mass
                "wrist_roll":           50.0,   # Low mass to move
            },
            damping={
                "shoulder_rotation":    80.0,
                "shoulder_pitch":       65.0,
                "elbow":                45.0,
                "wrist_pitch":          30.0,
                "wrist_roll":           20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr = ["gripper"],
            effort_limit_sim =          2.5,       # Increased from 1.9 to 2.5 for stronger grip
            velocity_limit_sim =        1.5,
            stiffness =                 60.0,      # Increased from 25.0 to 60.0 for more reliable closing
            damping =                   20.0,      # Increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of SO-ARM robot arm."""

# Removed FRANKA_PANDA_HIGH_PD_CFG as it's not applicable
