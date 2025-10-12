# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


# from isaaclab.utils.offset import OffsetCfg
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from isaaclab.utils.visualizer import FRAME_MARKER_CFG
# from isaaclab.utils.assets import RigidBodyPropertiesCfg


# Import both lift mdp and custom PicknPlace mdp
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from . import mdp as picknplace_mdp
from .config import PICKNPLACE_CFG

# Import from lift module
from SO_100.SO_100.tasks.lift.lift_env_cfg import SoArm100CubeCubeLiftEnvCfg
from SO_100.robots import SO_ARM100_CFG 

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    # target platform: platform on table for pick & place
    target_platform: RigidObjectCfg | None = None

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=PICKNPLACE_CFG.get_table_position(), rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.2, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Lift base rewards
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.04}, weight=2.0) #std was 0.05 and weight was 1.0
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=25.0) #weight was 15.0
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.2, "minimal_height": 0.04, "command_name": "object_pose"}, #std was 0.3 and weight was 16.0
        weight=12.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.04, "minimal_height": 0.04, "command_name": "object_pose"}, #std was 0.05 and weight was 5.0
        weight=8.0,
    )

    # PicknPlace specific rewards
    transporting_object = RewTerm(
        func=picknplace_mdp.distance_to_target_area_reward, 
        params={"std": 0.05, "minimal_height": 0.04},
        weight=10.0
    )
    
    placing_object = RewTerm(
        func=picknplace_mdp.object_on_target_area_reward,
        params={},  # Uses central config
        weight=25.0
    )
    
    object_drop_penalty = RewTerm(
        func=picknplace_mdp.object_drop_penalty,
        params={"drop_height_threshold": 0.005},
        weight=-10.0
    )

    # Additional rewards
    gripper_alignment = RewTerm(
        func=picknplace_mdp.gripper_object_alignment,
        params={"std": 0.02},
        weight=2.0
    )
    
    # Gripper release after placement
    gripper_release_after_placement = RewTerm(
        func=picknplace_mdp.gripper_release_after_placement,
        params={},
        weight=5.0  # Moderate weight to encourage release
    )
    
    object_dragging_penalty = RewTerm(
        func=picknplace_mdp.object_dragging_penalty,
        params={"minimal_height": 0.04, "velocity_threshold": 0.1},
        weight=-15.0
    )

    # Penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-05) #weight was -1e-4
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-05, #weight was -1e-4
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP - PicknPlace version."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=picknplace_mdp.object_dropping, 
        params={"drop_height_threshold": 0.005}
    )
    
    placement_success = DoneTerm(
        func=picknplace_mdp.object_placement_success,
        params={}  # Uses central config
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# PicknPlace Environments - Enhanced from original lift for pick and place tasks
##

@configclass
class SoArm100CubePicknPlaceEnvCfg(SoArm100CubeCubeLiftEnvCfg):
    """PicknPlace environment - enhanced from original lift with target placement functionality"""
    def __post_init__(self):
        # post init of parent (inherits all lift configuration including correct gripper settings)
        super().__post_init__()
        
        # Update object position to use PicknPlace config
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=PICKNPLACE_CFG.get_cube_position(), rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.3, 0.3, 0.3),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        
        # Camera position
        self.viewer.eye = (2.5, 2.5, 1.5)
        
        # Add small platform for target placement
        self.scene.target_platform = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetPlatform",
            init_state=RigidObjectCfg.InitialStateCfg(pos=PICKNPLACE_CFG.get_platform_position(), rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=PICKNPLACE_CFG.PLATFORM_SIZE,  # Using central config for size
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,  # Make it immovable
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,  # PhysX requires > 0
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),  # Enable collision
                mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.4, 0.2),  # Wooden brown color like a small table
                    metallic=0.0,
                    roughness=0.7,
                ),
            ),
        )
        
        # Extend episode length for more complex task
        self.episode_length_s = 10.0


@configclass
class SoArm100CubePicknPlaceEnvCfg_PLAY(SoArm100CubePicknPlaceEnvCfg):
    """PicknPlace play environment - adapted from training environment"""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

