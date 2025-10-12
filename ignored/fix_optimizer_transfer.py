#!/usr/bin/env python3
"""
Fix optimizer state for V0->V1 transfer
Remove Adam optimizer momentum/variance states to prevent tensor size mismatch
"""

import torch
import os

def clean_optimizer_state(model_path: str):
    """Remove optimizer state to prevent tensor size conflicts"""
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Keep model weights, remove optimizer state
    cleaned_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        # Remove these to force fresh optimizer initialization:
        # 'optimizer_state_dict': checkpoint.get('optimizer_state_dict', {}),
        # 'lr_scheduler_state_dict': checkpoint.get('lr_scheduler_state_dict', {}),
    }
    
    # Keep other metadata if exists
    for key in ['iteration', 'epoch', 'best_reward']:
        if key in checkpoint:
            cleaned_checkpoint[key] = checkpoint[key]
    
    # Save cleaned version
    backup_path = model_path.replace('.pt', '_with_optimizer.pt')
    torch.save(checkpoint, backup_path)
    print(f"✅ Original saved as backup: {backup_path}")
    
    torch.save(cleaned_checkpoint, model_path)
    print(f"✅ Cleaned model saved: {model_path}")
    print("🔧 Optimizer state removed - will initialize fresh")

if __name__ == "__main__":
    model_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-19_transfer/model_335597.pt"
    clean_optimizer_state(model_path)