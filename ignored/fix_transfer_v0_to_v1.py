#!/usr/bin/env python3
"""
CORRECTED Transfer V0 PicknPlace model to V1 
Problem: Previous transfer mapped wrong observation dimensions
Solution: Preserve core motor+object observations, add new dual cube data at the end
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path

def corrected_transfer_v0_to_v1(v0_model_path: str, v1_save_path: str):
    """
    OPTIMIZED V0 -> V1 Transfer Strategy:
    
    V0 (28-dim): [joint_pos(6), joint_vel(6), object_pos(3), target_pos(7), actions(6)]
    V1 (38-dim): [joint_pos(6), joint_vel(6), object_pos(3), target_pos(7), actions(6), cube2_pos(3), stack_target(7)]
    
    Strategy: Keep V0 core observations at SAME positions, add minimal new V1 data
    """
    print(f"Loading V0 model from: {v0_model_path}")
    v0_checkpoint = torch.load(v0_model_path, map_location='cpu')
    v0_state_dict = v0_checkpoint['model_state_dict']
    
    # Create new state dict for V1
    v1_state_dict = {}
    
    # V0 -> V1 Observation Mapping Strategy:
    # Keep first 28 dims identical to V0, add 10 new dims for V1 features
    
    for key, tensor in v0_state_dict.items():
        if key == 'actor.0.weight':
            print(f"FINAL Actor expansion: {tensor.shape} -> [256, 38]")
            old_weight = tensor  # [256, 28]
            new_weight = torch.zeros(256, 38)
            
            # PRESERVE V0 connections exactly (dims 0-27)
            new_weight[:, :28] = old_weight
            
            # Initialize NEW V1 connections (dims 28-37) with proper initialization
            # Use smaller initialization for new dual cube features
            torch.nn.init.xavier_uniform_(new_weight[:, 28:], gain=0.01)
            
            v1_state_dict[key] = new_weight
            
        elif key == 'critic.0.weight':
            print(f"FINAL Critic expansion: {tensor.shape} -> [256, 38]")
            old_weight = tensor  # [256, 28] 
            new_weight = torch.zeros(256, 38)
            
            # PRESERVE V0 connections exactly (dims 0-27)
            new_weight[:, :28] = old_weight
            
            # Initialize NEW V1 connections (dims 28-37) with proper initialization
            torch.nn.init.xavier_uniform_(new_weight[:, 28:], gain=0.01)
            
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
    
    print(f"✅ FINAL V1 model saved to: {v1_save_path}")
    print("🎯 V0 core motor+object skills preserved at SAME dimensions!")
    print("🆕 New V1 dual cube features added at dimensions 28-37")
    
    return v1_save_path

if __name__ == "__main__":
    # Best V0 model
    v0_model = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace/2025-08-19_13-19-01/model_335597.pt"
    
    # Create FINAL V1 directory with correct 38-dim 
    v1_dir = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-20_final_transfer_38dim"
    v1_model = f"{v1_dir}/model_335597.pt"
    
    # Perform CORRECTED transfer
    transferred_model = corrected_transfer_v0_to_v1(v0_model, v1_model)
    
    print("\n🚀 Use FINAL transfer model:")
    print(f"--load_run=2025-08-20_final_transfer_38dim --checkpoint=model_335597.pt")