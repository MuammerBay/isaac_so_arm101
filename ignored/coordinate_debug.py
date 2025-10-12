#!/usr/bin/env python3
"""
Coordinate system debug - understand why robot is going to wrong place
"""

def analyze_coordinate_issue():
    """Analyze the coordinate system mismatch."""
    
    print("="*70)
    print("COORDINATE SYSTEM ANALYSIS")
    print("="*70)
    
    # Current configuration
    robot_pos = [0.0, 0.0, 0.0]
    table_pos = [0.5, 0.0, 0.0]
    cube1_spawn = [0.12, 0.05, 0.015]
    cube2_spawn = [0.15, 0.08, 0.015]
    
    # Targets
    command_target = [-0.15, -0.25, 0.043]     # What command system shows
    stack_base_target = [0.25, -0.15, 0.0575] # Where reward system wants cube1
    target_area = [0.25, -0.15, 0.043]        # Target area position
    
    print("WORLD POSITIONS:")
    print(f"Robot:           {robot_pos}")
    print(f"Table:           {table_pos}")  
    print(f"Cube1 spawn:     {cube1_spawn}")
    print(f"Cube2 spawn:     {cube2_spawn}")
    print()
    print("TARGET POSITIONS:")
    print(f"Command target:  {command_target}")
    print(f"Stack base:      {stack_base_target}")
    print(f"Target area:     {target_area}")
    
    print("\\n" + "="*70)
    print("HYPOTHESIS: Command is in ROBOT FRAME, rewards in WORLD FRAME")
    print("="*70)
    
    # Convert command target from robot frame to world frame
    # Assuming simple translation (robot at origin)
    command_world = [robot_pos[0] + command_target[0],
                     robot_pos[1] + command_target[1], 
                     robot_pos[2] + command_target[2]]
    
    print(f"Command target in world frame: {command_world}")
    
    # Check if this makes sense
    import numpy as np
    
    # Distance from cube1 to command target (world frame)
    cube1_to_command_world = np.linalg.norm(np.array(cube1_spawn) - np.array(command_world))
    cube1_to_stack_base = np.linalg.norm(np.array(cube1_spawn) - np.array(stack_base_target))
    
    print(f"\\nDISTANCE ANALYSIS:")
    print(f"Cube1 → Command (world): {cube1_to_command_world:.3f}m")
    print(f"Cube1 → Stack base:      {cube1_to_stack_base:.3f}m")
    
    print(f"\\nRECOMMENDATION:")
    print("❌ DON'T change command coordinates - they may be in robot frame")
    print("✅ Command system and reward system use different coordinate frames")
    print("✅ Robot learns from REWARDS, not commands")
    print("✅ The 'alakasız nokta' problem suggests deeper issue")
    
    print(f"\\n" + "="*70)
    print("DEBUGGING STEPS:")
    print("="*70)
    print("1. Check if reward functions are working correctly")
    print("2. Verify phase detection is working")  
    print("3. Check if cube1_placement_gated is being called")
    print("4. Monitor actual reward values during training")
    print("5. Verify STACK_BASE_POS is correct in world coordinates")

if __name__ == "__main__":
    analyze_coordinate_issue()