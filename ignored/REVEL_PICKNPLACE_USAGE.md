# Revel PicknPlace Task Usage Guide

## Overview
`revel_picknplace-v0` is a cloned and independent task derived from `picknplace-v0`. This task provides complete isolation with separate logging, configurations, and training runs.

## Task Information
- **Training Task ID**: `SO-ARM100-Lift-Cube-RevelPicknplace-v0`
- **Play Task ID**: `SO-ARM100-Lift-Cube-RevelPicknplace-Play-v0`
- **Log Directory**: `so_arm100_revel_picknplace`
- **Base Architecture**: Inherits from V1 lift configuration

## Training Commands

### Basic Training
```bash
python scripts/rsl_rl/train.py --task=SO-ARM100-Lift-Cube-RevelPicknplace-v0
```

### Training with Custom Parameters
```bash
python scripts/rsl_rl/train.py \
    --task=SO-ARM100-Lift-Cube-RevelPicknplace-v0 \
    --num_envs=2048 \
    --max_iterations=1500 \
    --headless
```

### Training with Tensorboard Logging
```bash
python scripts/rsl_rl/train.py \
    --task=SO-ARM100-Lift-Cube-RevelPicknplace-v0 \
    --num_envs=2048 \
    --headless \
    --logger=tensorboard
```

## Play Commands

### Basic Play (Latest Model)
```bash
python scripts/rsl_rl/play.py \
    --task=SO-ARM100-Lift-Cube-RevelPicknplace-Play-v0 \
    --num_envs=32
```

### Play with Specific Model
```bash
python scripts/rsl_rl/play.py \
    --task=SO-ARM100-Lift-Cube-RevelPicknplace-Play-v0 \
    --num_envs=32 \
    --checkpoint=logs/rsl_rl/so_arm100_revel_picknplace/YYYY-MM-DD_HH-MM-SS/model_XXXX.pt
```

### Play with Video Recording
```bash
python scripts/rsl_rl/play.py \
    --task=SO-ARM100-Lift-Cube-RevelPicknplace-Play-v0 \
    --num_envs=16 \
    --enable_cameras \
    --video \
    --video_length=200
```

## Directory Structure

### Training Logs Location
```
logs/rsl_rl/so_arm100_revel_picknplace/
├── YYYY-MM-DD_HH-MM-SS/          # Timestamp folder
│   ├── model_XXXX.pt              # Saved models
│   ├── config.yaml                # Training configuration
│   └── events.out.tfevents.*      # Tensorboard logs
```

### Task Source Code
```
source/SO_100/SO_100/tasks/revel_picknplace/
├── config.py                      # Task configuration
├── revel_picknplace_env_cfg.py    # Environment configuration
├── __init__.py                    # Task registration
├── agents/
│   └── rsl_rl_ppo_cfg.py         # PPO configuration
└── mdp/
    ├── observations.py
    ├── rewards.py
    └── terminations.py
```

## Configuration Details

### Environment Configuration
- **Class**: `SoArm100CubeRevelPicknPlaceEnvCfg`
- **Play Class**: `SoArm100CubeRevelPicknPlaceEnvCfg_PLAY`
- **Config Instance**: `REVEL_PICKNPLACE_CFG`

### Agent Configuration
- **Class**: `RevelPicknPlaceCubePPORunnerCfg`
- **Max Iterations**: 1500
- **Steps per Environment**: 24
- **Save Interval**: 50 iterations

### Default Training Parameters
- **Actor/Critic Hidden Dims**: [256, 128, 64]
- **Activation**: ELU
- **Initial Noise Std**: 1.0
- **Empirical Normalization**: False

## Monitoring Training

### Tensorboard (if enabled)
```bash
tensorboard --logdir=logs/rsl_rl/so_arm100_revel_picknplace
```

### Check Latest Model
```bash
ls -la logs/rsl_rl/so_arm100_revel_picknplace/*/model_*.pt | tail -1
```

## Task Independence

This task is completely independent from the original `picknplace-v0`:
- ✅ Separate log directories
- ✅ Independent configuration classes
- ✅ Isolated MDP components
- ✅ Unique task registration
- ✅ No shared state or dependencies

## Troubleshooting

### Common Issues
1. **Task not found**: Ensure the task is properly registered by checking `__init__.py`
2. **Import errors**: Verify all config references use `REVEL_PICKNPLACE_CFG`
3. **Log conflicts**: Logs are stored in separate `so_arm100_revel_picknplace` directory

### Verification Commands
```bash
# Check task registration
python -c "import gym; print([env for env in gym.envs.registry.env_specs.keys() if 'RevelPicknplace' in env])"

# Verify configuration
python -c "from source.SO_100.SO_100.tasks.revel_picknplace.config import REVEL_PICKNPLACE_CFG; print('Config loaded successfully')"
```

## Next Steps

1. **Training**: Start with basic training command
2. **Monitoring**: Use Tensorboard to track progress
3. **Testing**: Use play commands to evaluate trained models
4. **Customization**: Modify task configuration as needed for your specific requirements

---

**Note**: This task maintains the same core functionality as the original picknplace-v0 but operates in complete isolation, allowing for independent development and experimentation.