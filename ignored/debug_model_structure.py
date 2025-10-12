#!/usr/bin/env python3
"""
Debug PyTorch model structure to understand the exact format
"""

import zipfile
import pickle
import io
import struct

def debug_pytorch_model(model_path):
    """Debug PyTorch model internal structure."""
    print(f"=== Debugging {model_path} ===")
    
    with zipfile.ZipFile(model_path, 'r') as zf:
        print(f"Files in ZIP: {zf.namelist()}")
        
        # Find the data.pkl file
        pkl_files = [f for f in zf.namelist() if f.endswith('data.pkl')]
        print(f"Pickle files: {pkl_files}")
        
        if pkl_files:
            pkl_file = pkl_files[0]
            print(f"\nReading {pkl_file}...")
            
            try:
                with zf.open(pkl_file) as f:
                    # Try to load pickle data
                    data = pickle.load(f)
                    print(f"Pickle data type: {type(data)}")
                    
                    if hasattr(data, '__dict__'):
                        print(f"Attributes: {data.__dict__.keys()}")
                    elif isinstance(data, dict):
                        print(f"Dict keys: {list(data.keys())}")
                        
                    # Look for tensor references
                    def find_tensor_refs(obj, path=""):
                        if hasattr(obj, '__dict__'):
                            for key, value in obj.__dict__.items():
                                find_tensor_refs(value, f"{path}.{key}")
                        elif isinstance(obj, dict):
                            for key, value in obj.items():
                                if isinstance(value, str) and 'storage' in value.lower():
                                    print(f"  Tensor reference at {path}[{key}]: {value}")
                                find_tensor_refs(value, f"{path}[{key}]")
                        elif isinstance(obj, (list, tuple)):
                            for i, value in enumerate(obj):
                                find_tensor_refs(value, f"{path}[{i}]")
                    
                    find_tensor_refs(data)
                        
            except Exception as e:
                print(f"Error reading pickle: {e}")
                
                # Try raw binary reading
                with zf.open(pkl_file) as f:
                    raw_data = f.read()
                    print(f"Raw data size: {len(raw_data)} bytes")
                    print(f"First 100 bytes: {raw_data[:100]}")

def compare_model_structures():
    """Compare original and expanded model structures."""
    
    original_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-19_13-19-01/model_335597.pt"
    expanded_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-19_13-19-01/model_335597_expanded.pt"
    target_path = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/model_335600.pt"
    
    print("COMPARING MODEL STRUCTURES")
    print("="*50)
    
    debug_pytorch_model(original_path)
    print("\n" + "="*50)
    debug_pytorch_model(expanded_path) 
    print("\n" + "="*50)
    debug_pytorch_model(target_path)

if __name__ == "__main__":
    compare_model_structures()