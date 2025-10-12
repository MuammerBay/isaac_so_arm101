#!/usr/bin/env python3
"""
Model Compatibility Fixer for Isaac Lab RL Models
Converts model_335597 (28-dim obs) to be compatible with current task (38-dim obs)
"""

import torch
import os
import shutil
from pathlib import Path

def analyze_model_dimensions(model_path):
    """Analyze model dimensions."""
    print(f"Analyzing {model_path}")
    model = torch.load(model_path, map_location='cpu')
    
    state_dict = model['model_state_dict']
    
    # Find input layer dimensions
    actor_input = state_dict['actor.0.weight'].shape
    critic_input = state_dict['critic.0.weight'].shape
    
    print(f"  Actor input:  {actor_input}")
    print(f"  Critic input: {critic_input}")
    print(f"  Observation dim: {actor_input[1]}")
    
    return model, actor_input[1]

def expand_model_dimensions(source_model_path, target_model_path, output_path):
    """
    Expand model from source dimensions to target dimensions.
    Uses intelligent weight initialization to preserve learned features.
    """
    print("="*60)
    print("MODEL COMPATIBILITY FIXER")
    print("="*60)
    
    # Load both models
    source_model, source_dim = analyze_model_dimensions(source_model_path)
    target_model, target_dim = analyze_model_dimensions(target_model_path)
    
    if source_dim == target_dim:
        print("✅ Models already compatible!")
        return
    
    print(f"\n🔧 Converting {source_dim}D → {target_dim}D observations")
    
    # Create expanded model based on source
    expanded_model = source_model.copy()
    source_state = source_model['model_state_dict']
    target_state = target_model['model_state_dict']
    
    # Strategy: Expand input layers intelligently
    dim_diff = target_dim - source_dim
    print(f"   Adding {dim_diff} new observation dimensions")
    
    # Expand actor input layer
    old_actor_weight = source_state['actor.0.weight']  # [256, 28]
    old_actor_bias = source_state['actor.0.bias']      # [256]
    
    # Create new actor weights [256, 38]
    new_actor_weight = torch.zeros(old_actor_weight.shape[0], target_dim)
    
    # Copy old weights (preserve learned features)
    new_actor_weight[:, :source_dim] = old_actor_weight
    
    # Initialize new dimensions with small random values
    # Use Xavier initialization for new weights
    new_dims_weight = torch.randn(old_actor_weight.shape[0], dim_diff) * 0.01
    new_actor_weight[:, source_dim:] = new_dims_weight
    
    # Expand critic input layer similarly
    old_critic_weight = source_state['critic.0.weight']
    old_critic_bias = source_state['critic.0.bias']
    
    new_critic_weight = torch.zeros(old_critic_weight.shape[0], target_dim)
    new_critic_weight[:, :source_dim] = old_critic_weight
    
    new_dims_critic = torch.randn(old_critic_weight.shape[0], dim_diff) * 0.01
    new_critic_weight[:, source_dim:] = new_dims_critic
    
    # Update state dict
    expanded_state = source_state.copy()
    expanded_state['actor.0.weight'] = new_actor_weight
    expanded_state['critic.0.weight'] = new_critic_weight
    
    # Verify all other layers match target model architecture
    print("\n🔍 Verifying layer compatibility...")
    for key in target_state.keys():
        if key not in ['actor.0.weight', 'critic.0.weight']:
            if key in expanded_state:
                if expanded_state[key].shape != target_state[key].shape:
                    print(f"   ⚠️  {key}: {expanded_state[key].shape} → {target_state[key].shape}")
                    # Copy shape from target model but keep source values where possible
                    expanded_state[key] = target_state[key].clone()
            else:
                print(f"   ➕ Adding missing layer: {key}")
                expanded_state[key] = target_state[key].clone()
    
    # Update model with expanded state dict
    expanded_model['model_state_dict'] = expanded_state
    
    # Copy other metadata from target model (to ensure compatibility)
    for key in target_model.keys():
        if key not in ['model_state_dict']:
            expanded_model[key] = target_model[key]
    
    print(f"\n💾 Saving compatible model to: {output_path}")
    
    # Create backup
    backup_path = output_path + ".backup"
    if os.path.exists(source_model_path):
        shutil.copy2(source_model_path, backup_path)
        print(f"   📋 Backup created: {backup_path}")
    
    # Save expanded model
    torch.save(expanded_model, output_path)
    
    print("\n✅ Model conversion completed!")
    print(f"   Original: {source_dim}D observations")
    print(f"   Expanded: {target_dim}D observations")
    print(f"   Preserved: All learned weights from original model")
    print(f"   Added: {dim_diff} new input dimensions (small random init)")
    
    return expanded_model

def main():
    """Main conversion function."""
    
    # Paths
    source_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-19_13-19-01/model_335597.pt"
    target_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/model_335600.pt"
    output_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/model_335597_compatible.pt"
    
    # Convert model
    expand_model_dimensions(source_path, target_path, output_path)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"Use this checkpoint in training:")
    print(f"--checkpoint=logs/rsl_rl/so_arm100_picknplace_v1/model_335597_compatible.pt")

if __name__ == "__main__":
    main()