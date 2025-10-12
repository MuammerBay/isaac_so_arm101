# V0 to V1 Model Transfer Guide

## 🎯 Goal
Continue training V1 (with gripper fix) from your successful V0 model to leverage learned arm movement skills.

---

## ✅ Compatibility Check

**CONFIRMED COMPATIBLE**:
- ✅ Same network architecture: [256, 128, 64]  
- ✅ Same action space: 5 arm + 1 gripper
- ✅ Same observation space
- ✅ Only gripper logic differs (fixed in V1)

---

## 🚀 Method 1: Direct Cross-Experiment Resume (RECOMMENDED)

### Step 1: Find Your Best V0 Model
```bash
# List all V0 runs
ls -la logs/rsl_rl/so_arm100_lift/

# Check models in your best run (example: 2025-08-08_19-04-36)
ls logs/rsl_rl/so_arm100_lift/2025-08-08_19-04-36/ | grep model | tail -5

# Result should show: model_99200.pt (or similar high number)
```

### Step 2: Start V1 Training from V0 Model
```bash
# Resume V1 from V0's best model
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-v1 --headless \
    --resume \
    --load_run logs/rsl_rl/so_arm100_lift/2025-08-08_19-04-36 \
    --load_checkpoint model_99200 \
    --max_iterations 3000
```

### Step 3: Monitor Transfer Learning
```bash
# Monitor V1 training progress
tensorboard --logdir logs/rsl_rl/so_arm100_lift_v1
```

**Expected Behavior**:
- ✅ Should start with good arm movement (from V0)
- ✅ Should quickly learn gripper usage (V1 fix)
- ✅ Should achieve lifting faster than training from scratch

---

## 🚀 Method 2: Manual Model Transfer

### Step 1: Create Transfer Directory
```bash
# Create timestamped transfer directory
TRANSFER_DIR="logs/rsl_rl/so_arm100_lift_v1/from_v0_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$TRANSFER_DIR"
echo "Created: $TRANSFER_DIR"
```

### Step 2: Copy V0 Model and Configuration
```bash
# Copy the model
cp logs/rsl_rl/so_arm100_lift/2025-08-08_19-04-36/model_99200.pt \
   "$TRANSFER_DIR/model_0.pt"

# Copy configuration for reference
cp -r logs/rsl_rl/so_arm100_lift/2025-08-08_19-04-36/params \
      "$TRANSFER_DIR/"

# Verify copy
ls -la "$TRANSFER_DIR"
```

### Step 3: Resume Training
```bash
# Start training from copied model
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-v1 --headless \
    --resume --load_run $(basename "$TRANSFER_DIR") \
    --max_iterations 3000
```

---

## 🚀 Method 3: Checkpoint-to-Checkpoint Transfer

### Advanced: Start V1 from Specific V0 Checkpoint
```bash
# Resume from earlier V0 checkpoint (not final model)
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-v1 --headless \
    --resume \
    --load_run logs/rsl_rl/so_arm100_lift/2025-08-08_19-04-36 \
    --load_checkpoint model_50000 \
    --max_iterations 5000

# This gives more room for V1 to adapt
```

---

## 📊 Expected Training Progression

### Phase 1: Model Loading (Iteration 0-50)
- ✅ Model loads successfully from V0
- ✅ Arm movement skills preserved
- ✅ Initial performance better than random

### Phase 2: Gripper Adaptation (Iteration 50-500)
- ✅ Robot learns fixed gripper values (0.0=open, 0.5=close)
- ✅ Grasping attempts begin
- ✅ Lifting rewards start appearing

### Phase 3: Integration (Iteration 500-1500)
- ✅ Coordinated reach + grasp + lift
- ✅ High success rates
- ✅ Performance exceeds original V0

### Phase 4: Fine-tuning (Iteration 1500+)
- ✅ Optimal policies
- ✅ Smooth trajectories
- ✅ Robust manipulation

---

## 📈 Success Metrics to Monitor

### TensorBoard Tracking
```bash
tensorboard --logdir logs/rsl_rl/so_arm100_lift_v1
```

### Key Metrics:
1. **Episode Reward**: Should start high (from V0 arm skills)
2. **Lifting Object Reward**: Should grow quickly (V1 gripper fix)
3. **Action Rate Penalty**: Should remain stable
4. **Policy Loss**: Should converge faster than training from scratch

### Visual Confirmation:
```bash
# Test with play script after 500 iterations
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Play-v1 \
    --load_run YOUR_TRANSFER_RUN --load_checkpoint model_500
```

---

## 🛠️ Troubleshooting

### Issue: Model Won't Load
```bash
# Check file paths
ls -la logs/rsl_rl/so_arm100_lift/2025-08-08_19-04-36/
ls -la logs/rsl_rl/so_arm100_lift_v1/

# Verify model file exists
file logs/rsl_rl/so_arm100_lift/2025-08-08_19-04-36/model_99200.pt
```

### Issue: Poor Initial Performance
```bash
# Try different V0 checkpoint
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-v1 --headless \
    --resume --load_run 2025-08-08_19-04-36 \
    --load_checkpoint model_75000  # Earlier checkpoint
```

### Issue: Gripper Still Not Working
```bash
# Test gripper fix with random agent first
python scripts/random_agent.py --task SO-ARM100-Lift-Cube-v1

# Check if gripper moves in GUI
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-v1 --max_iterations 10
```

---

## ⚡ Quick Start Commands

### Option A: Simple Transfer (Most Common)
```bash
# One command to rule them all
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-v1 --headless \
    --resume --load_run 2025-08-08_19-04-36 --max_iterations 2000
```

### Option B: Safe Transfer (Create New Directory)
```bash
# Create transfer directory and copy model
TRANSFER_DIR="logs/rsl_rl/so_arm100_lift_v1/from_v0_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$TRANSFER_DIR"
cp logs/rsl_rl/so_arm100_lift/2025-08-08_19-04-36/model_99200.pt "$TRANSFER_DIR/model_0.pt"

# Resume training
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-v1 --headless \
    --resume --load_run $(basename "$TRANSFER_DIR") --max_iterations 2000
```

---

## 🎯 Benefits of Transfer Learning

1. **Faster Convergence**: Skip arm movement learning phase
2. **Better Sample Efficiency**: Leverage V0's 249K iterations  
3. **Focused Learning**: Only learn gripper coordination
4. **Risk Mitigation**: Keep V0 safe, experiment in V1
5. **Baseline Comparison**: Compare V1 vs V0 performance

---

## 📝 Best Practices

1. **Start with fewer iterations** (1000-2000) to test transfer
2. **Monitor early performance** - should be better than random
3. **Compare with V0** - V1 should eventually exceed V0
4. **Save checkpoints frequently** - resume if something breaks
5. **Use TensorBoard** - visualize the transfer learning curve

Transfer learning is perfect for your use case - you get the best of both worlds! 🚀