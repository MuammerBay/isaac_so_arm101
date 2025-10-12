#!/usr/bin/env python3
"""
Fix PyTorch model metadata (data.pkl) to reflect new tensor dimensions
"""

import zipfile
import struct
import tempfile
import os
import shutil

def create_corrected_model():
    """Create a properly corrected model by copying structure from target model."""
    
    print("="*60)
    print("PYTORCH MODEL STRUCTURE FIXER")
    print("="*60)
    
    source_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-19_13-19-01/model_335597.pt"
    target_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/model_335600.pt"
    output_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-19_13-19-01/model_335597_fixed.pt"
    
    print(f"Source (28D + training data): {os.path.basename(source_path)}")
    print(f"Target (38D structure):       {os.path.basename(target_path)}")
    print(f"Output (combined):            {os.path.basename(output_path)}")
    
    # Strategy: Use target model structure, but copy compatible tensors from source
    print("\\n🔧 Creating hybrid model with correct structure...")
    
    # Start by copying target model (correct structure)
    temp_path = output_path + ".tmp"
    shutil.copy2(target_path, temp_path)
    
    # Now we need to selectively replace tensors where dimensions match
    # and we want to preserve training progress
    
    print("\\n📊 Analyzing tensor compatibility...")
    
    # Get storage info from both models
    source_storages = {}
    target_storages = {}
    
    with zipfile.ZipFile(source_path, 'r') as source_zip:
        for filename in source_zip.namelist():
            if '/data/' in filename and filename.split('/')[-1].isdigit():
                storage_id = int(filename.split('/')[-1])
                data = source_zip.read(filename)
                source_storages[storage_id] = data
    
    with zipfile.ZipFile(target_path, 'r') as target_zip:
        for filename in target_zip.namelist():
            if '/data/' in filename and filename.split('/')[-1].isdigit():
                storage_id = int(filename.split('/')[-1])
                data = target_zip.read(filename)
                target_storages[storage_id] = data
    
    print(f"Source model storages: {len(source_storages)}")
    print(f"Target model storages: {len(target_storages)}")
    
    # Identify which storages we can copy from source (same size)
    compatible_storages = []
    incompatible_storages = []
    
    for storage_id in target_storages:
        if storage_id in source_storages:
            source_size = len(source_storages[storage_id])
            target_size = len(target_storages[storage_id])
            
            if source_size == target_size:
                compatible_storages.append(storage_id)
                print(f"  ✅ Storage {storage_id}: Compatible ({source_size} bytes)")
            else:
                incompatible_storages.append(storage_id)
                print(f"  ❌ Storage {storage_id}: Size mismatch ({source_size} vs {target_size} bytes)")
        else:
            print(f"  ➕ Storage {storage_id}: Only in target model")
    
    print(f"\\nCompatible storages (will copy from source): {len(compatible_storages)}")
    print(f"Incompatible storages (will keep target): {len(incompatible_storages)}")
    
    # Now rebuild the model with selective tensor replacement
    final_path = output_path + ".final"
    
    with zipfile.ZipFile(temp_path, 'r') as template_zip:
        with zipfile.ZipFile(final_path, 'w', compression=zipfile.ZIP_STORED) as output_zip:
            
            for item in template_zip.infolist():
                data = template_zip.read(item.filename)
                
                # Check if this is a tensor storage we want to replace
                if '/data/' in item.filename and item.filename.split('/')[-1].isdigit():
                    storage_id = int(item.filename.split('/')[-1])
                    
                    if storage_id in compatible_storages:
                        # Replace with source data (preserve training)
                        source_data = source_storages[storage_id]
                        output_zip.writestr(item.filename, source_data)
                        print(f"  🔄 Replaced storage {storage_id} with source data")
                    else:
                        # Keep target data (correct dimensions)
                        output_zip.writestr(item.filename, data)
                        print(f"  📋 Kept storage {storage_id} from target (incompatible)")
                else:
                    # Copy all other files from target (metadata, structure, etc.)
                    output_zip.writestr(item.filename, data)
    
    # Replace temp with final
    os.replace(final_path, output_path)
    os.remove(temp_path)
    
    print("\\n✅ Hybrid model created successfully!")
    print("\\n📋 Model characteristics:")
    print("  - Structure: From target model (38D compatible)")
    print(f"  - Preserved tensors: {len(compatible_storages)} (training progress)")
    print(f"  - New tensors: {len(incompatible_storages)} (correct dimensions)")
    print("  - Metadata: From target model (compatibility)")
    
    print("\\n🧪 Test the fixed model:")
    print("python scripts/rsl_rl/train.py --task=SO-ARM100-Lift-Cube-Picknplace-v1 \\\\")
    print("  --checkpoint=model_335597_fixed.pt \\\\")
    print("  --resume --num_envs=4096 --max_iterations=10000 --headless --logger=tensorboard")
    
    return output_path

if __name__ == "__main__":
    create_corrected_model()