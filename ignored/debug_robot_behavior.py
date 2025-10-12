#!/usr/bin/env python3
"""
Debug why robot is taking cube backwards instead of to platform
"""

def debug_robot_movement():
    """Debug robot's movement logic."""
    
    print("="*70)
    print("ROBOT MOVEMENT DEBUG")
    print("="*70)
    
    # Current positions (back to original)
    robot_pos = [0.0, 0.0, 0.0]
    cube1_pos = [0.12, 0.05, 0.015]
    platform_pos = [0.25, -0.15, 0.032]
    target_pos = [0.25, -0.15, 0.0575]  # Where cube1 should go
    
    print(f"Robot position:    {robot_pos}")
    print(f"Cube1 position:    {cube1_pos}")  
    print(f"Platform position: {platform_pos}")
    print(f"Target position:   {target_pos}")
    
    # Calculate directions
    cube_to_target = [target_pos[0] - cube1_pos[0], 
                      target_pos[1] - cube1_pos[1],
                      target_pos[2] - cube1_pos[2]]
    
    print(f"\\nCube1 → Target direction: {cube_to_target}")
    print(f"  X: {cube_to_target[0]:+.3f} ({'forward' if cube_to_target[0] > 0 else 'backward'})")
    print(f"  Y: {cube_to_target[1]:+.3f} ({'right' if cube_to_target[1] > 0 else 'left'})")
    print(f"  Z: {cube_to_target[2]:+.3f} ({'up' if cube_to_target[2] > 0 else 'down'})")
    
    import numpy as np
    distance = np.linalg.norm(cube_to_target)
    print(f"\\nDistance to target: {distance:.3f}m")
    
    print(f"\\n" + "="*70)
    print("PROBLEM ANALYSIS")
    print("="*70)
    
    print("If robot is going 'backward' (negative Y direction), possible causes:")
    print("1. ❌ Phase detection: Robot not in PHASE_1_PLACE_CUBE1")
    print("2. ❌ Reward gating: cube1_placement_gated not active")  
    print("3. ❌ Observation space: Wrong target in observation")
    print("4. ❌ Neural network: Learned wrong association")
    
    print(f"\\n" + "="*70)
    print("DEBUGGING CHECKLIST")
    print("="*70)
    print("✅ Positions reverted to original")
    print("✅ Command system unchanged") 
    print("❓ Check training logs:")
    print("   - Is Episode_Reward/cube1_placement > 0?")
    print("   - Is robot in correct phase?")
    print("   - Are phase transitions working?")
    
    print(f"\\n" + "="*70)
    print("HYPOTHESIS")
    print("="*70)
    print("Robot may be following COMMAND target instead of REWARD target")
    print("Command target: (-0.15, -0.25) ← This is backward/left!")
    print("Reward target:  (0.25, -0.15)  ← This is forward/left")
    print("\\nIf robot goes to (-0.15, -0.25), it's following command, not rewards!")

if __name__ == "__main__":
    debug_robot_movement()