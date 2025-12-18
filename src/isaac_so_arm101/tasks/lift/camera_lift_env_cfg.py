# Copyright (c) 2024-2025, Vic Wu
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg

# from isaaclab.managers NotImplementedError
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaac_so_arm101.robots import SO_ARM100_CFG, SO_ARM100_ROSCON_CFG  # noqa: F401
from isaac_so_arm101.tasks.lift.joint_pos_env_cfg import SoArm100LiftCubeEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import CameraCfg
import isaaclab.sim as sim_utils


@configclass
class CameraRewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.01}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CameraObservationsCfg:
    """Override observations to add image features."""
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        #object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        camera_image_features = ObsTerm(
            func=mdp_obs.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )

        # 决定是否 concat（一般保持 true）
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class CameraEventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1), 
                "y": (-0.2, 0.2), 
                "z": (0.0, 0.0),
                # "roll": (-3.14159, 3.14159),    
                # "pitch": (-3.14159, 3.14159),    
                # "yaw": (-3.14159, 3.14159)      
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class SoArm100CameraLiftCubeEnvCfg(SoArm100LiftCubeEnvCfg):
    def __post_init__(self):
        """Currently identical to the base variant but defined separately for camera tasks."""
        super().__post_init__()
        self.scene.num_envs = 70
        
        self.scene.robot = SO_ARM100_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=SO_ARM100_CFG.init_state.replace(
                joint_pos={
                    "Shoulder_Rotation":    0.0,
                    "Shoulder_Pitch":       0.5,
                    "Elbow":                0.5,
                    "Wrist_Pitch":          0.5,
                    "Wrist_Roll": 0.0,
                    "Gripper": 0.3,  # Middle position to make movement more apparent
                }
            ),
        )

        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Fixed_Gripper/Camera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=("rgb",),
            offset=CameraCfg.OffsetCfg(
                pos=[0.0, -0.1, 0.0],
                rot=[0.96593, -0.25882, 0.0, 0.0],
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=4.0, horizontal_aperture=32.0, clipping_range=(0.01, 1.0e5)
            ),
        )
        self.observations = CameraObservationsCfg()
        self.rewards = CameraRewardsCfg()
        self.commands.object_pose.debug_vis=False
        self.events = CameraEventCfg()



@configclass
class SoArm100CameraLiftCubeEnvCfg_PLAY(SoArm100CameraLiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
