# SO-ARM100 Pick & Place Training Guide

*Comprehensive guide for training, resuming, playing, and customizing the PicknPlace task*

## 📋 Quick Reference

### Available PicknPlace Tasks
- `SO-ARM100-Lift-Cube-Picknplace-v0` - Training environment
- `SO-ARM100-Lift-Cube-Picknplace-Play-v0` - Evaluation/play environment

### Log Structure
- **V0 Logs**: `logs/rsl_rl/so_arm100_lift/`
- **V1 Logs**: `logs/rsl_rl/so_arm100_lift_v1/`
- **PicknPlace Logs**: `logs/rsl_rl/so_arm100_picknplace/` (completely separate)

---

## 🚀 1. Basic Training Commands

### Start New PicknPlace Training
```bash
# Basic training (1500 iterations)
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless

# Custom iteration count
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless --max_iterations 8000

# Training with custom environment count
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless --num_envs 2048

# Training with custom seed
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless --seed 42

# Training without headless (with GUI)
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0
```

### Start Training with Video Recording
```bash
# Record videos every 2000 steps, 200 frame length
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --video --video_interval 2000 --video_length 200

# Custom video settings
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --video --video_interval 1000 --video_length 300 --max_iterations 5000
```

---

## 🔄 2. Resume Training Commands

### Resume Latest Checkpoint
```bash
# Find your run directory first
ls logs/rsl_rl/so_arm100_picknplace/

# Resume with specific run name
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --resume --load_run 2025-08-16_15-30-45

# Resume with specific checkpoint number
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --resume --load_run 2025-08-16_15-30-45 --load_checkpoint 1000

# Resume with more iterations
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --resume --load_run 2025-08-16_15-30-45 --max_iterations 10000
```

### Find Available Checkpoints
```bash
# List all PicknPlace training runs
ls -la logs/rsl_rl/so_arm100_picknplace/

# List checkpoints in specific run
ls logs/rsl_rl/so_arm100_picknplace/2025-08-16_15-30-45/ | grep model

# Show latest checkpoint
ls logs/rsl_rl/so_arm100_picknplace/2025-08-16_15-30-45/ | grep model | tail -1
```

---

## 🎮 3. Play/Evaluation Commands

### Basic Play
```bash
# Play with latest checkpoint from specific run
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run 2025-08-16_15-30-45

# Play with specific checkpoint
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run 2025-08-16_15-30-45 --load_checkpoint 2000

# Play with custom number of environments
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run 2025-08-16_15-30-45 --num_envs 16
```

### Play with Video Recording
```bash
# Record play session
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run 2025-08-16_15-30-45 --video --video_length 500

# Record multiple episodes
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run 2025-08-16_15-30-45 --video --video_length 500 --num_envs 4
```

---

## 🎯 4. Environment Configuration

### Modify Environment Count in Code
Edit `source/SO_100/SO_100/tasks/lift/lift_env_cfg.py`:

```python
@configclass
class SoArm100CubeCubePicknPlaceEnvCfg(SoArm100CubeCubeLiftEnvCfg_v1):
    """PicknPlace environment - identical to V1 lift environment for now"""
    def __post_init__(self):
        # post init of parent (inherits all V1 fixes)
        super().__post_init__()
        
        # Custom environment count
        self.scene.num_envs = 1024  # Change from default 4096
        self.scene.env_spacing = 3.0  # Increase spacing if needed
```

### Modify Play Environment Count
```python
@configclass
class SoArm100CubeCubePicknPlaceEnvCfg_PLAY(SoArm100CubeCubePicknPlaceEnvCfg):
    """PicknPlace play environment - identical to V1 play environment"""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 25  # Change from default 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

---

## 📊 5. TensorBoard Logging

### Start TensorBoard
```bash
# Monitor PicknPlace training progress
tensorboard --logdir logs/rsl_rl/so_arm100_picknplace

# Monitor specific run
tensorboard --logdir logs/rsl_rl/so_arm100_picknplace/2025-08-16_15-30-45

# Compare V0 vs V1 vs PicknPlace
tensorboard --logdir logs/rsl_rl/so_arm100_lift --port 6006 &
tensorboard --logdir logs/rsl_rl/so_arm100_lift_v1 --port 6007 &
tensorboard --logdir logs/rsl_rl/so_arm100_picknplace --port 6008

# Monitor all experiments together
tensorboard --logdir logs/rsl_rl/
```

### Access TensorBoard
- Local: `http://localhost:6006`, `http://localhost:6007`, `http://localhost:6008`
- Remote: `http://YOUR_IP:6006`, `http://YOUR_IP:6007`, `http://YOUR_IP:6008`

---

## 🛠️ 6. Custom Training Parameters

### Modify PPO Parameters
Edit `source/SO_100/SO_100/tasks/lift/agents/rsl_rl_ppo_cfg.py`:

```python
@configclass 
class PicknPlaceCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PicknPlace PPO config - identical to V1 but with separate experiment name for isolated logging"""
    num_steps_per_env = 24
    max_iterations = 8000  # Increase iterations
    save_interval = 100    # Save more frequently
    experiment_name = "so_arm100_picknplace"
    empirical_normalization = False
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],  # Larger network
        critic_hidden_dims=[512, 256, 128], # Larger network
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,        # More exploration
        num_learning_epochs=8,    # More epochs
        num_mini_batches=8,       # More mini-batches
        learning_rate=5.0e-5,     # Lower learning rate
        schedule="adaptive",
        gamma=0.99,               # Different discount
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

---

## 🎨 7. Reward Customization

### Add Custom Rewards for Pick & Place
Edit `source/SO_100/SO_100/tasks/lift/lift_env_cfg.py`:

```python
@configclass
class SoArm100CubeCubePicknPlaceEnvCfg(SoArm100CubeCubeLiftEnvCfg_v1):
    """PicknPlace environment - identical to V1 lift environment for now"""
    def __post_init__(self):
        # post init of parent (inherits all V1 fixes)
        super().__post_init__()
        
        # FUTURE: Customize rewards for pick and place behavior
        # self.rewards.reaching_object.weight = 2.0  # Picking phase
        # self.rewards.lifting_object.weight = 15.0  # Lifting phase
        # self.rewards.placing_object = RewTerm(     # Placing phase
        #     func=mdp.object_placement_reward, weight=20.0
        # )
        
        # Modify action penalties for complex pick-place sequences
        self.rewards.action_rate.weight = -2e-5   # Adjusted penalty
```

---

## 🔍 8. Debugging and Testing

### Quick Functionality Test
```bash
# Test with zero agent (should load environment)
python scripts/zero_agent.py --task SO-ARM100-Lift-Cube-Picknplace-v0

# Test with random agent
python scripts/random_agent.py --task SO-ARM100-Lift-Cube-Picknplace-v0

# Short training test (10 iterations)
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless --max_iterations 10
```

### Check GPU Usage
```bash
# Monitor GPU during training
watch -n 1 nvidia-smi

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🔄 9. Transfer Learning from V1

### Start PicknPlace from V1 Model
```bash
# Use successful V1 model as starting point
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --resume \
    --load_run logs/rsl_rl/so_arm100_lift_v1/2025-08-15_14-31-07 \
    --load_checkpoint model_155000 \
    --max_iterations 3000

# Transfer from V1's best performing model
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --resume \
    --load_run logs/rsl_rl/so_arm100_lift_v1/2025-08-15_14-31-07 \
    --load_checkpoint model_158000 \
    --max_iterations 5000
```

**Expected Behavior:**
- ✅ Should start with good arm movement and gripper skills from V1
- ✅ Should adapt quickly to any pick-place specific modifications
- ✅ Should achieve good performance faster than training from scratch

---

## 📹 10. Video Recording Settings

### Video Recording Options
```bash
# High quality videos
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 \
    --video --video_interval 500 --video_length 400

# Play session recording
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run RUN_NAME --video --video_length 600

# Multiple camera angles (if supported)
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run RUN_NAME --video --enable_cameras
```

### Video Output Locations
- Training videos: `logs/rsl_rl/so_arm100_picknplace/RUN_NAME/videos/train/`
- Play videos: `logs/rsl_rl/so_arm100_picknplace/RUN_NAME/videos/play/`

---

## 🚨 11. Distributed Training

### Multi-GPU Training
```bash
# Distributed training on multiple GPUs
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --distributed --max_iterations 8000

# Specify GPU devices
CUDA_VISIBLE_DEVICES=0,1 python scripts/rsl_rl/train.py \
    --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless --distributed
```

---

## 📊 12. Performance Monitoring

### Check Training Progress
```bash
# Real-time log monitoring
tail -f logs/rsl_rl/so_arm100_picknplace/LATEST_RUN/train.log

# Check latest rewards
grep "reward" logs/rsl_rl/so_arm100_picknplace/LATEST_RUN/train.log | tail -10

# Monitor checkpoint creation
watch -n 10 "ls -la logs/rsl_rl/so_arm100_picknplace/LATEST_RUN/ | grep model"
```

### Performance Benchmarking
```bash
# Time training performance
time python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 \
    --headless --max_iterations 100

# Memory usage monitoring
python -c "
import psutil
import time
while True:
    print(f'RAM: {psutil.virtual_memory().percent}% CPU: {psutil.cpu_percent()}%')
    time.sleep(5)
" &
```

---

## 🔄 13. Experiment Management

### Organized Experiment Workflow
```bash
# 1. Start experiment with descriptive run name
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --max_iterations 5000 --run_name "baseline_test"

# 2. Monitor with TensorBoard
tensorboard --logdir logs/rsl_rl/so_arm100_picknplace &

# 3. Test intermediate checkpoints
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run TIMESTAMP_baseline_test --load_checkpoint 2000

# 4. Resume if needed
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --resume --load_run TIMESTAMP_baseline_test --max_iterations 8000
```

---

## 🎯 14. Success Metrics

### Evaluate Model Performance
```bash
# Play with metrics
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run YOUR_RUN --num_envs 100  # Test on many environments

# Record success rate
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run YOUR_RUN --video --video_length 1000  # Long episode
```

### Compare Models
```bash
# Compare different checkpoints
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run YOUR_RUN --load_checkpoint 1000

python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run YOUR_RUN --load_checkpoint 3000

python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 \
    --load_run YOUR_RUN --load_checkpoint 5000
```

---

## 🛠️ 15. Troubleshooting

### Common Issues and Solutions

#### Out of Memory
```bash
# Reduce environment count
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --num_envs 1024  # Instead of 4096

# Or modify in code (see section 4)
```

#### Slow Training
```bash
# Enable distributed training
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --distributed

# Check GPU utilization
nvidia-smi -l 1
```

#### Failed Resume
```bash
# Check available checkpoints
ls logs/rsl_rl/so_arm100_picknplace/YOUR_RUN/ | grep model

# Make sure run name is correct
ls logs/rsl_rl/so_arm100_picknplace/
```

---

## 📚 16. Quick Commands Cheat Sheet

```bash
# TRAINING
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless --max_iterations 8000
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless --video

# RESUME
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless --resume --load_run RUN_NAME

# PLAY
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 --load_run RUN_NAME
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0 --load_run RUN_NAME --video

# MONITOR
tensorboard --logdir logs/rsl_rl/so_arm100_picknplace
ls logs/rsl_rl/so_arm100_picknplace/
tail -f logs/rsl_rl/so_arm100_picknplace/LATEST_RUN/*.log

# TEST
python scripts/zero_agent.py --task SO-ARM100-Lift-Cube-Picknplace-v0
python scripts/random_agent.py --task SO-ARM100-Lift-Cube-Picknplace-v0

# TRANSFER FROM V1
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 --headless \
    --resume --load_run logs/rsl_rl/so_arm100_lift_v1/2025-08-15_14-31-07 \
    --load_checkpoint model_155000
```

---

## 🎯 17. Key Features

### Inherited from V1
1. **Fixed Gripper Controls**: Open = 0.0, Close = 0.5 (working correctly)
2. **Optimized Rewards**: Balanced reaching, lifting, and tracking rewards
3. **Stable Training**: Reduced action penalties for smooth learning
4. **Same Network Architecture**: [256, 128, 64] proven to work well

### PicknPlace Specific
1. **Isolated Logging**: `experiment_name = "so_arm100_picknplace"`
2. **Separate Log Directory**: `logs/rsl_rl/so_arm100_picknplace/`
3. **Transfer Learning Ready**: Can start from successful V1 models
4. **Customization Ready**: Environment prepared for pick-place modifications

---

## 🎯 Remember

1. **PicknPlace is completely isolated** from V0 and V1 - safe to experiment
2. **experiment_name = "so_arm100_picknplace"** ensures separate logging
3. **Always use TensorBoard** to monitor training progress
4. **Test with play script** to verify model performance
5. **Save intermediate checkpoints** with `save_interval` setting
6. **Can transfer learn from V1** for faster convergence
7. **All V1 gripper fixes included** - should work out of the box

Happy experimenting with PicknPlace! 🚀