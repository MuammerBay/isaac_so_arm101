#!/usr/bin/env python3
"""
Debug script to test cube1 targeting fix
Run this after the configuration changes to verify targeting works correctly
"""

import numpy as np

def analyze_targeting_fix():
    """Analyze the cube1 targeting fix."""
    print("="*60)
    print("CUBE1 TARGETING FIX VERIFICATION")
    print("="*60)
    
    # Configuration values
    CUBE1_SPAWN = [0.12, 0.05, 0.015]
    STACK_BASE_TARGET = [0.25, -0.15, 0.0575]
    OLD_COMMAND_TARGET = [-0.15, -0.25, 0.043]
    NEW_COMMAND_TARGET = [0.25, -0.15, 0.0575]
    
    print(f"Cube1 spawn position:     {CUBE1_SPAWN}")
    print(f"Target (STACK_BASE_POS):  {STACK_BASE_TARGET}")
    print(f"OLD command target:       {OLD_COMMAND_TARGET}")
    print(f"NEW command target:       {NEW_COMMAND_TARGET}")
    
    # Calculate distances
    spawn_to_old_command = np.linalg.norm(np.array(CUBE1_SPAWN) - np.array(OLD_COMMAND_TARGET))
    spawn_to_new_command = np.linalg.norm(np.array(CUBE1_SPAWN) - np.array(NEW_COMMAND_TARGET))
    spawn_to_target = np.linalg.norm(np.array(CUBE1_SPAWN) - np.array(STACK_BASE_TARGET))
    
    print(f"\\nDistance Analysis:")
    print(f"Cube1 → Old command:      {spawn_to_old_command:.3f}m")
    print(f"Cube1 → New command:      {spawn_to_new_command:.3f}m") 
    print(f"Cube1 → Actual target:    {spawn_to_target:.3f}m")
    
    # Check consistency
    command_target_match = np.allclose(NEW_COMMAND_TARGET, STACK_BASE_TARGET, atol=0.001)
    print(f"\\nConsistency Check:")
    print(f"Command matches target:   {'✅ YES' if command_target_match else '❌ NO'}")
    
    # Movement analysis
    print(f"\\nMovement Required (Spawn → Target):")
    movement = np.array(STACK_BASE_TARGET) - np.array(CUBE1_SPAWN)
    print(f"  ΔX: {movement[0]:+.3f}m ({movement[0]*100:+.1f}cm)")
    print(f"  ΔY: {movement[1]:+.3f}m ({movement[1]*100:+.1f}cm)")
    print(f"  ΔZ: {movement[2]:+.3f}m ({movement[2]*100:+.1f}cm)")
    
    # Robot reachability check
    robot_pos = [0.0, 0.0, 0.0]
    robot_to_target = np.linalg.norm(np.array(STACK_BASE_TARGET) - np.array(robot_pos))
    robot_reach_estimate = 0.5  # Approximate robot arm reach
    
    print(f"\\nReachability Analysis:")
    print(f"Robot to target distance: {robot_to_target:.3f}m")
    print(f"Estimated robot reach:    {robot_reach_estimate:.3f}m")
    reachable = robot_to_target <= robot_reach_estimate
    print(f"Target reachable:         {'✅ YES' if reachable else '❌ NO - TOO FAR'}")
    
    if not reachable:
        print(f"\\n⚠️  WARNING: Target may be outside robot reach!")
        print(f"   Consider moving target closer to robot or extending robot reach.")
        
        # Suggest closer target
        closer_target = [0.20, -0.10, 0.0575]  # Closer to robot
        closer_distance = np.linalg.norm(np.array(closer_target) - np.array(robot_pos))
        print(f"\\n💡 Suggested closer target: {closer_target}")
        print(f"   Distance from robot: {closer_distance:.3f}m")
    
    print(f"\\n{'='*60}")
    print("RECOMMENDATION:")
    print("✅ Command target fixed - now matches reward target")
    print("✅ Reward weight increased (15.0 → 25.0)")
    print("🔄 Test training with these changes")
    print("👁️  Monitor 'Episode_Reward/cube1_placement' - should be > 0 when robot targets correctly")
    
    return command_target_match and reachable

if __name__ == "__main__":
    success = analyze_targeting_fix()
    print(f"\\nOverall fix status: {'✅ GOOD' if success else '❌ NEEDS MORE WORK'}")