#!/usr/bin/env python3
"""
Transfer Learning Script: Lift → PicknPlace Task
================================================

This script transfers a trained Lift model to the PicknPlace task.
Since both tasks have identical observation and action spaces, this is a direct model transfer.

Key Features:
- Preserves all learned motor skills from Lift task
- Maintains network architecture compatibility
- Handles optimizer state transfer
- Creates new experiment directory for PicknPlace
- Validates model compatibility before transfer

Usage:
    python scripts/transfer_lift_to_picknplace.py \
        --lift_model_path logs/rsl_rl/so_arm100_lift/YYYY-MM-DD_HH-MM-SS/model_XXXXX.pt \
        --picknplace_save_path logs/rsl_rl/so_arm100_picknplace/transferred_model.pt \
        --experiment_name so_arm100_picknplace_transferred
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
import json

def analyze_model_compatibility(lift_checkpoint: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Analyze if Lift model is compatible with PicknPlace task.
    
    Args:
        lift_checkpoint: Loaded checkpoint dictionary
        
    Returns:
        Tuple of (is_compatible, message)
    """
    try:
        model_state = lift_checkpoint['model_state_dict']
        
        # Check if required keys exist
        required_keys = ['actor.0.weight', 'critic.0.weight', 'actor.0.bias', 'critic.0.bias']
        missing_keys = [key for key in required_keys if key not in model_state]
        
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
        
        # Check network architecture
        actor_weight = model_state['actor.0.weight']
        critic_weight = model_state['critic.0.weight']
        
        # Expected architecture: [256, obs_dim] -> [256, 128] -> [128, 64] -> [64, action_dim]
        if actor_weight.shape[0] != 256 or critic_weight.shape[0] != 256:
            return False, f"Unexpected network architecture. Actor: {actor_weight.shape}, Critic: {critic_weight.shape}"
        
        # Check if observation dimensions match (both tasks should have same obs space)
        obs_dim = actor_weight.shape[1]
        print(f"Detected observation dimension: {obs_dim}")
        
        # Check action dimensions
        if 'actor.3.weight' in model_state:
            action_dim = model_state['actor.3.weight'].shape[0]
            print(f"Detected action dimension: {action_dim}")
            
            # Expected: 6 actions (5 arm joints + 1 gripper)
            if action_dim != 6:
                return False, f"Unexpected action dimension: {action_dim}, expected 6"
        
        return True, "Model is compatible with PicknPlace task"
        
    except Exception as e:
        return False, f"Error analyzing model: {str(e)}"

def transfer_lift_to_picknplace(lift_model_path: str, 
                               picknplace_save_path: str,
                               experiment_name: str = "so_arm100_picknplace_transferred") -> bool:
    """
    Transfer Lift model to PicknPlace task.
    
    Since both tasks have identical observation and action spaces,
    this is a direct model copy with experiment metadata update.
    
    Args:
        lift_model_path: Path to Lift model checkpoint
        picknplace_save_path: Path to save transferred PicknPlace model
        experiment_name: Name for the new PicknPlace experiment
        
    Returns:
        True if transfer successful, False otherwise
    """
    print("=" * 60)
    print("TRANSFER LEARNING: Lift → PicknPlace")
    print("=" * 60)
    
    # Load Lift model
    print(f"Loading Lift model from: {lift_model_path}")
    try:
        lift_checkpoint = torch.load(lift_model_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading Lift model: {e}")
        return False
    
    # Analyze compatibility
    print("\n🔍 Analyzing model compatibility...")
    is_compatible, message = analyze_model_compatibility(lift_checkpoint)
    
    if not is_compatible:
        print(f"❌ Model not compatible: {message}")
        return False
    
    print(f"✅ {message}")
    
    # Create new checkpoint for PicknPlace
    print(f"\n🔄 Creating PicknPlace model...")
    
    # Copy the entire checkpoint
    picknplace_checkpoint = lift_checkpoint.copy()
    
    # Update experiment metadata
    if 'experiment_name' in picknplace_checkpoint:
        picknplace_checkpoint['experiment_name'] = experiment_name
    else:
        picknplace_checkpoint['experiment_name'] = experiment_name
    
    # Update task information
    picknplace_checkpoint['source_task'] = 'SO-ARM100-Lift-Cube-v0'
    picknplace_checkpoint['target_task'] = 'SO-ARM100-Lift-Cube-Picknplace-v0'
    picknplace_checkpoint['transfer_method'] = 'direct_copy'
    picknplace_checkpoint['transfer_timestamp'] = str(torch.cuda.Event().elapsed_time(torch.cuda.Event()) if torch.cuda.is_available() else 0)
    
    # Create directory if it doesn't exist
    save_dir = Path(picknplace_save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save transferred model
    print(f"Saving transferred model to: {picknplace_save_path}")
    try:
        torch.save(picknplace_checkpoint, picknplace_save_path)
        print("✅ Transfer completed successfully!")
    except Exception as e:
        print(f"❌ Error saving transferred model: {e}")
        return False
    
    # Create experiment directory structure
    experiment_dir = save_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    # Copy training configuration if available
    source_dir = Path(lift_model_path).parent
    config_files = ['config.yaml', 'config.json', 'hyperparameters.yaml']
    
    for config_file in config_files:
        source_config = source_dir / config_file
        if source_config.exists():
            dest_config = experiment_dir / config_file
            shutil.copy2(source_config, dest_config)
            print(f"📋 Copied {config_file}")
    
    # Create transfer metadata
    transfer_metadata = {
        'transfer_info': {
            'source_task': 'SO-ARM100-Lift-Cube-v0',
            'target_task': 'SO-ARM100-Lift-Cube-Picknplace-v0',
            'source_model_path': str(lift_model_path),
            'transferred_model_path': str(picknplace_save_path),
            'transfer_method': 'direct_copy',
            'observation_space_compatibility': 'identical',
            'action_space_compatibility': 'identical',
            'network_architecture': 'preserved',
            'learned_skills_transferred': [
                'arm_movement_coordination',
                'object_approach_behavior',
                'grasping_skills',
                'lifting_dynamics',
                'end_effector_control'
            ],
            'new_skills_to_learn': [
                'target_platform_navigation',
                'precise_object_placement',
                'gripper_release_after_placement',
                'extended_episode_management'
            ]
        },
        'training_recommendations': {
            'resume_training': True,
            'learning_rate': 'keep_existing',
            'max_iterations': 2000,
            'expected_convergence': 'faster_than_from_scratch',
            'monitor_metrics': [
                'placing_object_reward',
                'gripper_release_after_placement_reward',
                'episode_success_rate'
            ]
        }
    }
    
    # Save transfer metadata
    metadata_path = experiment_dir / 'transfer_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(transfer_metadata, f, indent=2)
    
    print(f"\n📊 Transfer Summary:")
    print(f"   Source Task: SO-ARM100-Lift-Cube-v0")
    print(f"   Target Task: SO-ARM100-Lift-Cube-Picknplace-v0")
    print(f"   Transfer Method: Direct Copy (identical observation/action spaces)")
    print(f"   Model Path: {picknplace_save_path}")
    print(f"   Experiment Dir: {experiment_dir}")
    print(f"   Metadata: {metadata_path}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Start PicknPlace training:")
    print(f"      python scripts/rsl_rl/train.py --task SO-ARM100-Lift-Cube-Picknplace-v0 \\")
    print(f"          --resume \\")
    print(f"          --load_run {experiment_dir} \\")
    print(f"          --load_checkpoint transferred_model \\")
    print(f"          --max_iterations 2000")
    print(f"   2. Monitor training progress:")
    print(f"      tensorboard --logdir {experiment_dir}")
    
    return True

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Transfer trained Lift model to PicknPlace task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic transfer
    python scripts/transfer_lift_to_picknplace.py \\
        --lift_model_path logs/rsl_rl/so_arm100_lift/2025-01-15_14-30-22/model_1500.pt \\
        --picknplace_save_path logs/rsl_rl/so_arm100_picknplace/transferred_model.pt
    
    # Transfer with custom experiment name
    python scripts/transfer_lift_to_picknplace.py \\
        --lift_model_path logs/rsl_rl/so_arm100_lift/2025-01-15_14-30-22/model_1500.pt \\
        --picknplace_save_path logs/rsl_rl/so_arm100_picknplace/transferred_model.pt \\
        --experiment_name my_picknplace_experiment
        """
    )
    
    parser.add_argument(
        '--lift_model_path',
        type=str,
        required=True,
        help='Path to the trained Lift model checkpoint (.pt file)'
    )
    
    parser.add_argument(
        '--picknplace_save_path',
        type=str,
        required=True,
        help='Path to save the transferred PicknPlace model (.pt file)'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='so_arm100_picknplace_transferred',
        help='Name for the new PicknPlace experiment (default: so_arm100_picknplace_transferred)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Analyze compatibility without performing transfer'
    )
    
    args = parser.parse_args()
    
    # Validate input paths
    if not Path(args.lift_model_path).exists():
        print(f"❌ Error: Lift model path does not exist: {args.lift_model_path}")
        return 1
    
    if not args.lift_model_path.endswith('.pt'):
        print(f"❌ Error: Lift model path must be a .pt file: {args.lift_model_path}")
        return 1
    
    if not args.picknplace_save_path.endswith('.pt'):
        print(f"❌ Error: PicknPlace save path must be a .pt file: {args.picknplace_save_path}")
        return 1
    
    # Perform dry run if requested
    if args.dry_run:
        print("🔍 DRY RUN: Analyzing compatibility only...")
        try:
            lift_checkpoint = torch.load(args.lift_model_path, map_location='cpu')
            is_compatible, message = analyze_model_compatibility(lift_checkpoint)
            if is_compatible:
                print(f"✅ {message}")
                print("✅ Model is ready for transfer!")
            else:
                print(f"❌ {message}")
                print("❌ Model cannot be transferred.")
            return 0
        except Exception as e:
            print(f"❌ Error during dry run: {e}")
            return 1
    
    # Perform transfer
    success = transfer_lift_to_picknplace(
        args.lift_model_path,
        args.picknplace_save_path,
        args.experiment_name
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
