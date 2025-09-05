# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO-ARM100 5-DOF robot arm for livestream.

The following configurations are available:

* :obj:`SO_ARM100_CFG`: SO-ARM100 robot arm configuration.
* :obj:`SO_ARM100_ROS2_CFG`: SO-ARM100 robot arm configuration more adapted for sim2real.
        ->  converted from the xacro of this repository:
        https://github.com/JafarAbdi/ros2_so_arm100
"""

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
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Robots/so_arm100/so_100.usd",
        activate_contact_sensors=False,  # Adjust based on need
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(0.7071068, 0.0, 0.0, 0.7071068),  # Quaternion for 90 degrees rotation around Y-axis
        joint_pos={
            "Shoulder_Rotation": 0.0,
            "Shoulder_Pitch": 0.0,
            "Elbow": 0.0,
            "Wrist_Pitch": 0.0,
            "Wrist_Roll": 0.0,
            "Gripper": 0.3,  # Middle position to make movement more apparent
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Shoulder Pan      moves: ALL masses                   (~0.8kg total)
        # Shoulder Lift     moves: Everything except base       (~0.65kg)
        # Elbow             moves: Lower arm, wrist, gripper    (~0.38kg)
        # Wrist Pitch       moves: Wrist and gripper            (~0.24kg)
        # Wrist Roll        moves: Gripper assembly             (~0.14kg)
        # Jaw               moves: Only moving jaw              (~0.034kg)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Shoulder_.*", "Elbow", "Wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness={
                "Shoulder_Rotation": 200.0,  # Highest - moves all mass
                "Shoulder_Pitch": 170.0,  # Slightly less than rotation
                "Elbow": 120.0,  # Reduced based on less mass
                "Wrist_Pitch": 80.0,  # Reduced for less mass
                "Wrist_Roll": 50.0,  # Low mass to move
            },
            damping={
                "Shoulder_Rotation": 80.0,
                "Shoulder_Pitch": 65.0,
                "Elbow": 45.0,
                "Wrist_Pitch": 30.0,
                "Wrist_Roll": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Gripper"],
            effort_limit_sim=2.5,  # Increased from 1.9 to 2.5 for stronger grip
            velocity_limit_sim=1.5,
            stiffness=60.0,  # Increased from 25.0 to 60.0 for more reliable closing
            damping=20.0,  # Increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of SO-ARM robot arm."""

# Removed FRANKA_PANDA_HIGH_PD_CFG as it's not applicable


SO_ARM100_ROS2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Robots/so_arm100_ros2/so_arm100.usd",
        activate_contact_sensors=False,  # Adjust based on need
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(0.7071068, 0.0, 0.0, 0.7071068),  # Quaternion for 90 degrees rotation around Y-axis
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_pitch_joint": 0.0,
            "wrist_roll_joint": 0.0,
            "jaw_joint": 0.1,  # Middle position to make movement more apparent
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Shoulder Pan      moves: ALL masses                   (~0.8kg total)
        # Shoulder Lift     moves: Everything except base       (~0.65kg)
        # Elbow             moves: Lower arm, wrist, gripper    (~0.38kg)
        # Wrist Pitch       moves: Wrist and gripper            (~0.24kg)
        # Wrist Roll        moves: Gripper assembly             (~0.14kg)
        # Jaw               moves: Only moving jaw              (~0.034kg)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_joint", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness={
                "shoulder_pan_joint": 200.0,  # Highest - moves all mass
                "shoulder_lift_joint": 170.0,  # Slightly less than rotation
                "elbow_joint": 120.0,  # Reduced based on less mass
                "wrist_pitch_joint": 80.0,  # Reduced for less mass
                "wrist_roll_joint": 50.0,  # Low mass to move
            },
            damping={
                "shoulder_pan_joint": 80.0,
                "shoulder_lift_joint": 65.0,
                "elbow_joint": 45.0,
                "wrist_pitch_joint": 30.0,
                "wrist_roll_joint": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["jaw_joint"],
            effort_limit_sim=2.5,  # Increased from 1.9 to 2.5 for stronger grip
            velocity_limit_sim=1.5,
            stiffness=60.0,  # Increased from 25.0 to 60.0 for more reliable closing
            damping=20.0,  # Increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

"""Configuration of SO-ARM robot arm more adapted for sim2real."""
