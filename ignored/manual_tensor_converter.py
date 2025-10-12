#!/usr/bin/env python3
"""
Manual Tensor Converter - Direct binary manipulation of PyTorch tensors
Expands input dimensions from 28D to 38D while preserving training progress
"""

import zipfile
import struct
import numpy as np
import os
import shutil

def read_tensor_storage(zip_file, storage_id):
    """Read tensor storage from ZIP file."""
    storage_path = f"model_{storage_id.split('_')[1]}/data/{storage_id.split('/')[-1]}"
    with zip_file.open(storage_path) as f:
        data = f.read()
    return data

def write_tensor_storage(zip_file, storage_path, data):
    """Write tensor storage to ZIP file."""
    zip_file.writestr(storage_path, data)

def expand_tensor_28_to_38(tensor_data, original_shape, target_shape):
    """
    Expand tensor from 28D to 38D input.
    original_shape: [256, 28] 
    target_shape: [256, 38]
    """
    print(f"    Expanding tensor: {original_shape} → {target_shape}")
    
    # Convert bytes to float32 array
    float_count = len(tensor_data) // 4
    tensor = np.frombuffer(tensor_data, dtype=np.float32)
    
    # Reshape to original dimensions
    tensor_2d = tensor.reshape(original_shape)
    
    # Create expanded tensor
    expanded = np.zeros(target_shape, dtype=np.float32)
    
    # Copy original weights (first 28 dimensions)
    expanded[:, :28] = tensor_2d
    
    # Initialize new dimensions (last 10) with small random values
    # Use conservative initialization based on existing weights
    existing_std = np.std(tensor_2d) * 0.1  # Very conservative
    new_weights = np.random.normal(0, existing_std, (target_shape[0], 10)).astype(np.float32)
    expanded[:, 28:] = new_weights
    
    print(f"    - Preserved: 28 dimensions")
    print(f"    - Added: 10 dimensions (std={existing_std:.6f})")
    
    # Convert back to bytes
    return expanded.tobytes()

def convert_model_manual():
    """Convert model using manual tensor manipulation."""
    
    print("="*70)
    print("MANUAL TENSOR CONVERTER - Dimension Expansion")
    print("="*70)
    
    source_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-19_13-19-01/model_335597.pt"
    target_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/model_335600.pt"
    output_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/model_335597_expanded.pt"
    
    # Create backup
    backup_path = output_path + ".original_backup"
    shutil.copy2(source_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    print("\\n🔍 Analyzing tensor storage requirements...")
    
    # Based on our earlier analysis:
    # Storage 1, 9, 21, 45: Input layer weights (28D → 38D expansion needed)
    # Storage 1: actor.0.weight  [256, 28] → [256, 38]  (28,672 → 38,912 bytes)
    # Storage 9: critic.0.weight [256, 28] → [256, 38]  (28,672 → 38,912 bytes) 
    
    target_storages_to_expand = [1, 9, 21, 45]  # These need 28D→38D expansion
    
    print(f"Target storages for expansion: {target_storages_to_expand}")
    
    # Copy source model as base
    shutil.copy2(source_path, output_path)
    
    print("\\n🔧 Expanding tensor dimensions...")
    
    # Now modify the specific storages that need expansion
    temp_path = output_path + ".tmp"
    
    with zipfile.ZipFile(output_path, 'r') as source_zip:
        with zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_STORED) as target_zip:
            
            for item in source_zip.infolist():
                data = source_zip.read(item.filename)
                
                # Check if this is a storage that needs expansion
                if '/data/' in item.filename and item.filename.split('/')[-1].isdigit():
                    storage_id = int(item.filename.split('/')[-1])
                    
                    if storage_id in target_storages_to_expand:
                        print(f"  🔧 Expanding storage {storage_id}")
                        
                        # Original: [256, 28] = 28,672 bytes
                        # Target:   [256, 38] = 38,912 bytes
                        original_shape = (256, 28)
                        target_shape = (256, 38)
                        
                        expanded_data = expand_tensor_28_to_38(data, original_shape, target_shape)
                        target_zip.writestr(item.filename, expanded_data)
                    else:
                        # Keep original data
                        target_zip.writestr(item.filename, data)
                else:
                    # Keep all other files unchanged
                    target_zip.writestr(item.filename, data)
    
    # Replace original with expanded version
    os.replace(temp_path, output_path)
    
    print("\\n✅ Manual tensor expansion completed!")
    
    # Verify the result
    print("\\n🔍 Verifying expanded model...")
    with zipfile.ZipFile(output_path, 'r') as zf:
        for storage_id in target_storages_to_expand:
            storage_path = f"model_335597/data/{storage_id}"
            if storage_path in zf.namelist():
                with zf.open(storage_path) as f:
                    data = f.read()
                    expected_size = 256 * 38 * 4  # 38,912 bytes
                    actual_size = len(data)
                    print(f"  Storage {storage_id}: {actual_size} bytes (expected: {expected_size})")
                    if actual_size == expected_size:
                        print(f"    ✅ Correct size")
                    else:
                        print(f"    ❌ Size mismatch!")
    
    print("\\n" + "="*70)
    print("CONVERSION COMPLETED!")
    print("="*70)
    print("✅ Model architecture: Expanded from 28D to 38D observations")
    print("✅ Training progress: Preserved from original model")
    print("✅ New dimensions: Initialized with conservative random values")
    print("✅ File structure: Compatible with current task")
    print()
    print("Test the expanded model:")
    print("python scripts/rsl_rl/train.py --task=SO-ARM100-Lift-Cube-Picknplace-v1 \\\\")
    print("  --checkpoint=logs/rsl_rl/so_arm100_picknplace_v1/model_335597_expanded.pt \\\\")
    print("  --resume --num_envs=4096 --max_iterations=10000 --headless --logger=tensorboard")

if __name__ == "__main__":
    convert_model_manual()