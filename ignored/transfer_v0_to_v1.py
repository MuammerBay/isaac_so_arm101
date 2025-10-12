#!/usr/bin/env python3
"""
Transfer V0 PicknPlace model to V1 by adapting observation layer
Preserves all learned motor skills while expanding observation space
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path

def transfer_v0_to_v1_model(v0_model_path: str, v1_save_path: str):
    """
    Transfer V0 model (28-dim obs) to V1 (41-dim obs)
    Strategy: Expand input layer, preserve all other learned weights
    """
    print(f"Loading V0 model from: {v0_model_path}")
    v0_checkpoint = torch.load(v0_model_path, map_location='cpu')
    v0_state_dict = v0_checkpoint['model_state_dict']
    
    # Create new state dict for V1
    v1_state_dict = {}
    
    for key, tensor in v0_state_dict.items():
        if key == 'actor.0.weight':
            # Actor input layer: [256, 28] -> [256, 41]
            print(f"Expanding actor input: {tensor.shape} -> [256, 41]")
            old_weight = tensor  # [256, 28]
            new_weight = torch.zeros(256, 41)
            
            # Copy existing connections (first 28 dimensions)
            new_weight[:, :28] = old_weight
            
            # Initialize new connections (dim 28-40) with small random values
            torch.nn.init.xavier_uniform_(new_weight[:, 28:], gain=0.1)
            
            v1_state_dict[key] = new_weight
            
        elif key == 'critic.0.weight':
            # Critic input layer: [256, 28] -> [256, 41] 
            print(f"Expanding critic input: {tensor.shape} -> [256, 41]")
            old_weight = tensor  # [256, 28]
            new_weight = torch.zeros(256, 41)
            
            # Copy existing connections
            new_weight[:, :28] = old_weight
            
            # Initialize new connections with small random values
            torch.nn.init.xavier_uniform_(new_weight[:, 28:], gain=0.1)
            
            v1_state_dict[key] = new_weight
            
        else:
            # Keep all other layers unchanged (motor skills preserved)
            v1_state_dict[key] = tensor
    
    # Update checkpoint metadata
    v1_checkpoint = v0_checkpoint.copy()
    v1_checkpoint['model_state_dict'] = v1_state_dict
    
    # Save transferred model
    os.makedirs(os.path.dirname(v1_save_path), exist_ok=True)
    torch.save(v1_checkpoint, v1_save_path)
    
    print(f"✅ V1 model saved to: {v1_save_path}")
    print("🎯 Motor skills preserved, observation space expanded!")
    
    return v1_save_path

if __name__ == "__main__":
    # Best V0 model
    v0_model = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace/2025-08-19_13-19-01/model_335597.pt"
    
    # Create V1 directory and save transferred model
    v1_dir = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-19_transfer"
    v1_model = f"{v1_dir}/model_0.pt"
    
    # Perform transfer
    transferred_model = transfer_v0_to_v1_model(v0_model, v1_model)
    
    print("\n🚀 Now you can resume V1 training with:")
    print(f"python train.py task=SO-ARM100-Lift-Cube-Picknplace-v1 --resume={v1_dir}")