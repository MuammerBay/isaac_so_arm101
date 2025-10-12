# Transfer Learning Guide: Lift → PicknPlace

## 🎯 Overview

This guide explains how to transfer a trained Lift model to the PicknPlace task using the `transfer_lift_to_picknplace.py` script.

## 🔍 Task Analysis

### **Lift Task (Source)**
- **Goal:** Lift object above threshold
- **Episode Length:** 5.0 seconds
- **Observation Space:** 28 dimensions
  - `joint_pos` (6D): Joint positions
  - `joint_vel` (6D): Joint velocities  
  - `object_position` (3D): Object position in robot frame
  - `target_object_position` (7D): Target pose
  - `actions` (6D): Previous actions
- **Action Space:** 6 dimensions (5 arm joints + 1 gripper)
- **Rewards:** 4 base rewards (reaching, lifting, goal tracking, penalties)

### **PicknPlace Task (Target)**
- **Goal:** Pick object and place on target platform
- **Episode Length:** 10.0 seconds (extended)
- **Observation Space:** 28 dimensions (**IDENTICAL to Lift**)
  - Same observation structure as Lift
  - Inherits all Lift observations
- **Action Space:** 6 dimensions (**IDENTICAL to Lift**)
  - Same action structure as Lift
- **Rewards:** 4 base + 6 PicknPlace-specific rewards

### **Compatibility Analysis**
✅ **FULLY COMPATIBLE** - Both tasks have identical observation and action spaces!

## 🚀 Usage

### **Step 1: Find Your Best Lift Model**

```bash
# List all Lift training runs
ls -la logs/rsl_rl/so_arm100_lift/

# Check models in your best run (example: 2025-01-15_14-30-22)
ls logs/rsl_rl/so_arm100_lift/2025-01-15_14-30-22/ | grep model | tail -5

# Result should show: model_1500.pt (or similar high number)
```

### **Step 2: Run Transfer Script**

```bash
# Basic transfer
python scripts/transfer_lift_to_picknplace.py \
    --lift_model_path logs/rsl_rl/so_arm100_lift/2025-01-15_14-30-22/model_1500.pt \
    --picknplace_save_path logs/rsl_rl/so_arm100_picknplace/transferred_model.pt

# Transfer with custom experiment name
python scripts/transfer_lift_to_picknplace.py \
    --lift_model_path logs/rsl_rl/so_arm100_lift/2025-01-15_14-30-22/model_1500.pt \
    --picknplace_save_path logs/rsl_rl/so_arm100_picknplace/transferred_model.pt \
    --experiment_name my_picknplace_experiment
```

### **Step 3: Start PicknPlace Training**

```bash
# Resume PicknPlace training from transferred model
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 \
    --resume \
    --load_run logs/rsl_rl/so_arm100_picknplace \
    --load_checkpoint transferred_model \
    --max_iterations 2000
```

### **Step 4: Monitor Training**

```bash
# Monitor PicknPlace training progress
tensorboard --logdir logs/rsl_rl/so_arm100_picknplace
```

## 🔧 Script Options

### **Required Arguments**
- `--lift_model_path`: Path to trained Lift model (.pt file)
- `--picknplace_save_path`: Path to save transferred PicknPlace model (.pt file)

### **Optional Arguments**
- `--experiment_name`: Name for new PicknPlace experiment (default: `so_arm100_picknplace_transferred`)
- `--dry_run`: Analyze compatibility without performing transfer

### **Examples**

```bash
# Dry run - check compatibility only
python scripts/transfer_lift_to_picknplace.py \
    --lift_model_path logs/rsl_rl/so_arm100_lift/2025-01-15_14-30-22/model_1500.pt \
    --picknplace_save_path logs/rsl_rl/so_arm100_picknplace/transferred_model.pt \
    --dry_run

# Full transfer with custom experiment name
python scripts/transfer_lift_to_picknplace.py \
    --lift_model_path logs/rsl_rl/so_arm100_lift/2025-01-15_14-30-22/model_1500.pt \
    --picknplace_save_path logs/rsl_rl/so_arm100_picknplace/transferred_model.pt \
    --experiment_name my_custom_picknplace
```

## 📊 Expected Transfer Benefits

### **Preserved Skills from Lift Model:**
- ✅ **Arm Movement Coordination:** Learned joint control
- ✅ **Object Approach Behavior:** End-effector positioning
- ✅ **Grasping Skills:** Basic manipulation abilities
- ✅ **Lifting Dynamics:** Object handling experience
- ✅ **End-Effector Control:** Precise gripper positioning

### **New Skills to Learn in PicknPlace:**
- 🎯 **Target Platform Navigation:** Moving toward target area
- 🎯 **Precise Object Placement:** Positioning on target platform
- 🎯 **Gripper Release After Placement:** Letting go after success
- 🎯 **Extended Episode Management:** 10-second task completion
- 🎯 **PicknPlace-Specific Rewards:** New reward structure

## 🎯 Training Expectations

### **Early Training (0-500 iterations)**
- Robot should start with good arm movement (from Lift)
- May still hold object after placement (needs to learn release)
- Focus on learning PicknPlace-specific behaviors

### **Mid Training (500-1000 iterations)**
- Robot learns target platform navigation
- Begins to understand placement requirements
- Starts learning gripper release behavior

### **Advanced Training (1000-2000 iterations)**
- Robot masters complete pick-and-place sequence
- Consistent placement and release behavior
- High success rate with proper task completion

## 📈 Success Metrics

### **Key Metrics to Monitor:**
- `placing_object_reward`: Object on target area (weight: 25.0)
- `gripper_release_after_placement_reward`: Gripper release (weight: 5.0)
- `episode_success_rate`: Successful task completion
- `episode_length`: Should approach 10.0 seconds

### **Expected Performance:**
- **Faster Convergence:** 2-3x faster than training from scratch
- **Higher Success Rate:** Better initial performance
- **Stable Learning:** Smoother training curve
- **Better Final Performance:** Higher success rate

## 🚨 Troubleshooting

### **Common Issues:**

1. **Model Not Found:**
   ```bash
   # Check if model path exists
   ls -la logs/rsl_rl/so_arm100_lift/YYYY-MM-DD_HH-MM-SS/
   ```

2. **Incompatible Model:**
   ```bash
   # Run dry run to check compatibility
   python scripts/transfer_lift_to_picknplace.py --dry_run --lift_model_path <path>
   ```

3. **Training Fails to Resume:**
   ```bash
   # Check if transferred model exists
   ls -la logs/rsl_rl/so_arm100_picknplace/transferred_model.pt
   ```

### **Debug Commands:**

```bash
# Check model compatibility
python scripts/transfer_lift_to_picknplace.py --dry_run --lift_model_path <path>

# Verify transferred model
python -c "import torch; print(torch.load('<transferred_model_path>', map_location='cpu').keys())"

# Test environment loading
python -c "import gymnasium as gym; env = gym.make('SO-ARM100-Lift-Cube-Picknplace-v0'); print('Environment loaded successfully')"
```

## 📋 Transfer Process Summary

1. **✅ Analyze Compatibility:** Check if Lift model is compatible
2. **✅ Direct Model Copy:** Copy model weights (identical observation/action spaces)
3. **✅ Update Metadata:** Add transfer information to checkpoint
4. **✅ Create Experiment Structure:** Set up new experiment directory
5. **✅ Generate Transfer Report:** Create metadata with transfer details
6. **✅ Ready for Training:** Start PicknPlace training from transferred model

## 🎯 Expected Results

With proper transfer learning, you should see:
- **Immediate arm movement skills** from Lift model
- **Faster learning** of PicknPlace-specific behaviors
- **Higher success rate** compared to training from scratch
- **Smoother training curve** with fewer exploration phases
- **Better final performance** due to preserved motor skills

The transfer learning approach leverages all the hard-learned manipulation skills from the Lift task and applies them to the more complex PicknPlace task, significantly accelerating training and improving final performance!
