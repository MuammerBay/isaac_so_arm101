# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Register the SO-100 Cube PicknPlace environments (based on V1 Lift)
gym.register(
    id="SO-ARM100-Lift-Cube-Picknplace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.picknplace_env_cfg:SoArm100CubePicknPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PicknPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Lift-Cube-Picknplace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.picknplace_env_cfg:SoArm100CubePicknPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PicknPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)