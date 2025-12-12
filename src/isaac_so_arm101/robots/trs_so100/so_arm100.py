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

# Removed FRANKA_PANDA_HIGH_PD_CFG as it's not applicable here.
