# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, DeformableObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
from SO_100.robots import SO_ARM100_CFG
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

@configclass
class SO100DeformableCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Gripper"],
            open_command_expr={"Gripper": 0.5},
            close_command_expr={"Gripper": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "Fixed_Gripper"

        # Reduce the number of environments for training with deformables
        self.scene.num_envs = 1024
        # Disable replicator physics since deformable objects are not supported
        self.scene.replicate_physics = False

        # Reduce the PD for the hand to make the gripper less stiff
        # self.scene.robot.actuators["Fixed_Gripper"].stiffness = 200.0
        # self.scene.robot.actuators["Fixed_Gripper"].damping = 5.0

        # Set Deformable Cube as object
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.06, 0.06, 0.06),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                    self_collision_filter_distance=0.005,
                    settling_threshold=0.1,
                    sleep_damping=1.0,
                    sleep_threshold=0.05,
                    solver_position_iteration_count=20,
                    vertex_velocity_damping=0.5,
                    simulation_hexahedral_resolution=4,
                    rest_offset=0.0001,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(
                    dynamic_friction=0.95,
                    youngs_modulus=500000,
                ),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Fixed_Gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        # Set events for the specific object type (deformable cube)
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )


@configclass
class SO100DeformableCubeLiftEnvCfg_PLAY(SO100DeformableCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False