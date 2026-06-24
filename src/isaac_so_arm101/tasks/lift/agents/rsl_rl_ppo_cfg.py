# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO defaults for lift; conservative update vs template (stability + mild exploration)."""

    num_steps_per_env = 24
    # Longer training for pick-to-goal (lift) policies; override with --max_iterations if needed.
    max_iterations = 20000
    # Fewer disk writes than one save per 50 iters when max_iterations is large.
    save_interval = 500
    experiment_name = "lift"
    # Running obs normalization in RSL-RL often stabilizes value learning for mixed-scale observations.
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        # Lower starting exploration: precision placement needs the action noise to shrink, not grow.
        init_noise_std=0.8,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        # 0.01 drove action-noise std up to ~3.0 in long runs (no fine placement). Lower it for precision.
        entropy_coef=0.004,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=6.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
