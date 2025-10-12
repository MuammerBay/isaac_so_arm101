# Lift to PicknPlace Task Conversion Report

## Overview

This report documents the conversion process from the base **Lift task** to the enhanced **PicknPlace task** in the Isaac Lab SO-ARM100 robot simulation environment. The PicknPlace task extends the Lift task with additional complexity for pick-and-place manipulation scenarios.

## Task Comparison Summary

| Aspect | Lift Task | PicknPlace Task |
|--------|-----------|-----------------|
| **Primary Goal** | Lift object to target height | Pick object and place on target area |
| **Episode Length** | 5.0 seconds | 10.0 seconds |
| **Scene Complexity** | Robot + Object + Table | Robot + Object + Table + Platform + Target Area |
| **Reward Structure** | 4 base rewards | 4 base + 4 picknplace-specific rewards |
| **Termination Conditions** | 2 conditions | 4 conditions |
| **Target Definition** | Height-based | Position-based (target area) |

## Conversion Process

### 1. **Inheritance Structure**

The PicknPlace task is built using a multi-level inheritance approach:

```
LiftEnvCfg (Base)
    ↓
SoArm100CubeCubeLiftEnvCfg_v1 (V1 with gripper fixes)
    ↓
SoArm100CubePicknPlaceEnvCfg (PicknPlace implementation)
```

### 2. **Scene Configuration Extensions**

#### **Base Lift Scene:**
- Robot (SO-ARM100)
- Object (Cube)
- Table
- Lights

#### **PicknPlace Scene Additions:**
```python
# Additional scene elements in PicknPlace
self.scene.target_platform = RigidObjectCfg(...)  # Small platform on table
```

**New Scene Elements:**
- **Target Platform**: 8x8x2cm wooden platform positioned closer to robot
- **Target Area**: Invisible logical area defined by configuration (no visual marker)
- **Enhanced Positioning**: All elements positioned using central configuration

### 3. **Episode Length Extension**

```python
# Base Lift Task
self.episode_length_s = 5.0  # Default for lift tasks

# PicknPlace Task
self.episode_length_s = 10.0  # Extended for complex pick-and-place sequence
```

**Rationale**: PicknPlace requires more time for the complete sequence:
1. Reach object
2. Grasp object
3. Lift object
4. Transport to target
5. Place on target area

### 4. **Reward System Enhancement**

#### **Base Lift Rewards (4 components):**
```python
reaching_object = RewTerm(func=mdp.object_ee_distance, weight=2.0)
lifting_object = RewTerm(func=mdp.object_is_lifted, weight=25.0)
object_goal_tracking = RewTerm(func=mdp.object_goal_distance, weight=12.0)
object_goal_tracking_fine_grained = RewTerm(func=mdp.object_goal_distance, weight=8.0)
```

#### **PicknPlace Additional Rewards (4 components):**
```python
# Transport reward - guides object to target area
transporting_object = RewTerm(
    func=picknplace_mdp.distance_to_target_area_reward, 
    weight=10.0
)

# Placement reward - successful placement on target
placing_object = RewTerm(
    func=picknplace_mdp.object_on_target_area_reward,
    weight=25.0
)

# Drop penalty - prevents object dropping
object_drop_penalty = RewTerm(
    func=picknplace_mdp.object_drop_penalty,
    weight=-10.0
)

# Alignment reward - gripper-object alignment
gripper_alignment = RewTerm(
    func=picknplace_mdp.gripper_object_alignment,
    weight=2.0
)
```

### 5. **Termination Conditions**

#### **Base Lift Terminations (2 conditions):**
- Time limit (episode length)
- Object dropped below threshold

#### **PicknPlace Additional Terminations (2 new conditions):**
```python
# Success condition - object placed on target area
object_placement_success = DoneTerm(
    func=picknplace_mdp.object_placement_success,
    params={"threshold": 0.02}
)

# Drop condition - object dropped during transport
object_dropping = DoneTerm(
    func=picknplace_mdp.object_dropping,
    params={"drop_height_threshold": 0.005}
)
```

### 6. **Configuration Management**

#### **Central Configuration System**
The PicknPlace task introduces a centralized configuration class:

```python
class PicknPlaceConfig:
    # Robot configuration
    ROBOT_POS = [0.0, 0.0, 0.0]
    
    # Table configuration  
    TABLE_POS = [0.5, 0.0, 0.0]
    
    # Target platform configuration
    PLATFORM_POS = [0.25, -0.15, 0.032]
    PLATFORM_SIZE = [0.08, 0.08, 0.02]
    
    # Target area configuration
    TARGET_AREA_POS = [0.25, -0.15, 0.043]
    TARGET_AREA_SIZE = [0.05, 0.05, 0.001]
    
    # Pickup object configuration
    CUBE_POS = [0.2, 0.0, 0.015]
    CUBE_SIZE = [0.03, 0.03, 0.03]
    
    # Goal tolerances
    PLACEMENT_TOLERANCE_XY = 0.02
    PLACEMENT_TOLERANCE_Z = 0.015
    TERMINATION_TOLERANCE_XY = 0.015
    TERMINATION_TOLERANCE_Z = 0.01
```

**Benefits:**
- Single source of truth for all positions
- Easy parameter tuning
- Consistent positioning across components
- Utility methods for position calculations

### 7. **Custom MDP Components**

#### **New Reward Functions:**
- `distance_to_target_area_reward()` - Guides transport to target
- `object_on_target_area_reward()` - Rewards successful placement
- `object_drop_penalty()` - Penalizes dropping objects
- `gripper_object_alignment()` - Encourages proper grasping
- `object_dragging_penalty()` - Prevents dragging objects

#### **New Termination Functions:**
- `object_placement_success()` - Checks if object is on target area
- `object_dropping()` - Detects object drops during transport

#### **New Observation Functions:**
- `object_position_in_robot_root_frame()` - Object position relative to robot

### 8. **Training Configuration**

#### **PPO Hyperparameters:**
Both tasks use identical PPO configurations to ensure compatibility:

```python
# Identical to V1 Lift configuration
max_iterations = 1500
learning_rate = 1.0e-4
actor_hidden_dims = [256, 128, 64]
critic_hidden_dims = [256, 128, 64]
entropy_coef = 0.006
gamma = 0.98
```

#### **Experiment Naming:**
- Lift: `"so_arm100_lift"`
- PicknPlace: `"so_arm100_picknplace"`

## Key Technical Differences

### **1. Task Complexity**
- **Lift**: Single-phase task (grasp → lift)
- **PicknPlace**: Multi-phase task (grasp → lift → transport → place)

### **2. Spatial Requirements**
- **Lift**: Only requires vertical movement
- **PicknPlace**: Requires 3D navigation to target area

### **3. Success Criteria**
- **Lift**: Object height above threshold
- **PicknPlace**: Object position within target area bounds

### **4. Reward Engineering**
- **Lift**: Focus on lifting behavior
- **PicknPlace**: Balance between lifting, transport, and placement

### **5. Scene Geometry**
- **Lift**: Simple table setup
- **PicknPlace**: Complex scene with platform and target area

## Implementation Benefits

### **1. Modularity**
- Clean inheritance structure
- Reusable base components
- Easy to extend further

### **2. Configuration Management**
- Centralized parameter control
- Easy experimentation
- Consistent positioning

### **3. Reward Engineering**
- Phase-specific rewards
- Balanced reward structure
- Clear success criteria

### **4. Training Stability**
- Inherits proven V1 lift configuration
- Compatible hyperparameters
- Stable training process

## Usage Examples

### **Training PicknPlace:**
```bash
python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0
```

### **Playback PicknPlace:**
```bash
python scripts/rsl_rl/play.py --task SO-ARM100-Lift-Cube-Picknplace-Play-v0
```

### **Environment Listing:**
```bash
python scripts/list_envs.py
# Shows both lift and picknplace environments
```

## Conclusion

The PicknPlace task successfully extends the base Lift task with:

1. **Enhanced Scene Complexity** - Additional platform and target area
2. **Extended Episode Length** - More time for complex manipulation
3. **Sophisticated Reward Structure** - Multi-phase reward engineering
4. **Centralized Configuration** - Easy parameter management
5. **Modular Architecture** - Clean inheritance and extensibility

This conversion demonstrates a robust approach to building complex manipulation tasks on top of simpler base tasks, maintaining compatibility while adding significant functionality.

## Files Modified/Created

### **New Files:**
- `source/SO_100/SO_100/tasks/picknplace/__init__.py`
- `source/SO_100/SO_100/tasks/picknplace/picknplace_env_cfg.py`
- `source/SO_100/SO_100/tasks/picknplace/config.py`
- `source/SO_100/SO_100/tasks/picknplace/mdp/rewards.py`
- `source/SO_100/SO_100/tasks/picknplace/mdp/terminations.py`
- `source/SO_100/SO_100/tasks/picknplace/mdp/observations.py`
- `source/SO_100/SO_100/tasks/picknplace/agents/rsl_rl_ppo_cfg.py`

### **Base Files (Unchanged):**
- `source/SO_100/SO_100/tasks/lift/lift_env_cfg.py`
- `source/SO_100/SO_100/robots/so_arm100.py`

This architecture ensures that the base Lift task remains unchanged while providing a solid foundation for the more complex PicknPlace task.
