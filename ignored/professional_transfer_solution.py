#!/usr/bin/env python3
"""
PROFESSIONAL TRANSFER LEARNING SOLUTION: 28D → 38D 
Research-based weight expansion for Actor-Critic networks
===========================================================

Based on academic research and best practices:
1. Preserve learned V0 knowledge (28 dims)
2. Intelligently initialize new features (10 dims) 
3. Maintain training stability
4. Follow PyTorch transfer learning patterns
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Any

class NeuralNetworkExpander:
    """
    Professional neural network dimension expansion tool
    Based on transfer learning research and PyTorch best practices
    """
    
    def __init__(self):
        self.expansion_strategies = {
            'zero_init': self._zero_initialization,
            'small_random': self._small_random_initialization, 
            'kaiming_uniform': self._kaiming_uniform_initialization,
            'orthogonal': self._orthogonal_initialization,  # Best for RL
            'intelligent_copy': self._intelligent_copy_initialization
        }
    
    def expand_linear_layer_weights(self, 
                                  old_weight: torch.Tensor, 
                                  old_bias: torch.Tensor,
                                  new_input_dim: int,
                                  strategy: str = 'orthogonal') -> tuple:
        """
        Expand Linear layer from old_input_dim to new_input_dim
        Research shows orthogonal initialization is best for RL
        
        Args:
            old_weight: Shape [output_dim, old_input_dim] 
            old_bias: Shape [output_dim]
            new_input_dim: Target input dimension
            strategy: Initialization strategy for new weights
            
        Returns:
            (new_weight, new_bias): Expanded tensors
        """
        old_output_dim, old_input_dim = old_weight.shape
        additional_dims = new_input_dim - old_input_dim
        
        if additional_dims <= 0:
            raise ValueError(f"New dimension {new_input_dim} must be > old dimension {old_input_dim}")
        
        print(f"🔧 Expanding Linear layer: {old_input_dim} → {new_input_dim} (+{additional_dims} dims)")
        print(f"   Strategy: {strategy}")
        
        # Create new weight tensor
        new_weight = torch.zeros(old_output_dim, new_input_dim, dtype=old_weight.dtype)
        
        # Copy existing weights (preserve learned knowledge)
        new_weight[:, :old_input_dim] = old_weight
        
        # Initialize new dimensions
        additional_weights = self.expansion_strategies[strategy](
            old_output_dim, additional_dims, old_weight
        )
        new_weight[:, old_input_dim:] = additional_weights
        
        # Bias remains the same
        new_bias = old_bias.clone()
        
        return new_weight, new_bias
    
    def _zero_initialization(self, output_dim: int, input_dim: int, reference_weight: torch.Tensor) -> torch.Tensor:
        """Zero initialization - safe but may slow learning"""
        return torch.zeros(output_dim, input_dim, dtype=reference_weight.dtype)
    
    def _small_random_initialization(self, output_dim: int, input_dim: int, reference_weight: torch.Tensor) -> torch.Tensor:
        """Small random values - balanced approach"""
        std = reference_weight.std().item() * 0.1  # 10% of existing weight std
        return torch.normal(0, std, size=(output_dim, input_dim), dtype=reference_weight.dtype)
    
    def _kaiming_uniform_initialization(self, output_dim: int, input_dim: int, reference_weight: torch.Tensor) -> torch.Tensor:
        """Kaiming uniform - PyTorch default for Linear layers"""
        new_weights = torch.empty(output_dim, input_dim, dtype=reference_weight.dtype)
        nn.init.kaiming_uniform_(new_weights, a=np.sqrt(5))
        return new_weights
    
    def _orthogonal_initialization(self, output_dim: int, input_dim: int, reference_weight: torch.Tensor) -> torch.Tensor:
        """Orthogonal initialization - BEST for RL according to research"""
        new_weights = torch.empty(output_dim, input_dim, dtype=reference_weight.dtype)
        nn.init.orthogonal_(new_weights, gain=1.0)
        return new_weights
    
    def _intelligent_copy_initialization(self, output_dim: int, input_dim: int, reference_weight: torch.Tensor) -> torch.Tensor:
        """Intelligent copying - copy similar existing weights with noise"""
        old_input_dim = reference_weight.shape[1]
        
        # Strategy: Copy weights from similar dimensions with small random noise
        new_weights = torch.zeros(output_dim, input_dim, dtype=reference_weight.dtype)
        
        for i in range(input_dim):
            # Copy from corresponding position in old weights (with wraparound)
            source_idx = i % old_input_dim
            new_weights[:, i] = reference_weight[:, source_idx]
            
            # Add small noise to avoid exact duplication
            noise_std = reference_weight.std().item() * 0.05  # 5% noise
            noise = torch.normal(0, noise_std, size=(output_dim,), dtype=reference_weight.dtype)
            new_weights[:, i] += noise
        
        return new_weights

def create_v0_to_v1_transfer_model(v0_model_path: str, 
                                   output_path: str,
                                   expansion_strategy: str = 'orthogonal',
                                   preserve_optimizer: bool = False) -> str:
    """
    Create V1-compatible model from V0 model using professional transfer learning
    
    Args:
        v0_model_path: Path to V0 model (28D input)
        output_path: Path for new V1 model (38D input) 
        expansion_strategy: Weight initialization strategy for new dimensions
        preserve_optimizer: Whether to attempt optimizer state transfer
        
    Returns:
        Path to created V1 model
    """
    print("🎓 PROFESSIONAL V0 → V1 TRANSFER LEARNING")
    print("=" * 80)
    
    # Load V0 model
    print(f"📂 Loading V0 model: {v0_model_path}")
    v0_checkpoint = torch.load(v0_model_path, map_location='cpu')
    v0_model_state = v0_checkpoint['model_state_dict']
    
    # Initialize expander
    expander = NeuralNetworkExpander()
    
    # Create new model state dict
    v1_model_state = {}
    
    print(f"\n🔄 PROCESSING MODEL LAYERS:")
    
    for key, tensor in v0_model_state.items():
        if key in ['actor.0.weight', 'critic.0.weight']:
            # This is the first layer - needs dimension expansion
            network_type = 'Actor' if 'actor' in key else 'Critic'
            bias_key = key.replace('weight', 'bias')
            old_bias = v0_model_state[bias_key]
            
            print(f"\n🎯 Expanding {network_type} first layer:")
            print(f"   Original: {tensor.shape}")
            
            new_weight, new_bias = expander.expand_linear_layer_weights(
                tensor, old_bias, new_input_dim=38, strategy=expansion_strategy
            )
            
            print(f"   Expanded: {new_weight.shape}")
            
            v1_model_state[key] = new_weight
            v1_model_state[bias_key] = new_bias
            
        elif key in ['actor.0.bias', 'critic.0.bias']:
            # Already handled above
            continue
        else:
            # All other layers remain the same
            v1_model_state[key] = tensor.clone()
            print(f"   ♻️ Preserved: {key} -> {tensor.shape}")
    
    # Create new checkpoint
    v1_checkpoint = {
        'model_state_dict': v1_model_state,
        'iter': v0_checkpoint.get('iter', 0),
        'infos': {
            'transfer_learning': True,
            'source_model': v0_model_path,
            'expansion_strategy': expansion_strategy,
            'original_input_dim': 28,
            'new_input_dim': 38,
            'transfer_date': str(torch.datetime.datetime.now() if hasattr(torch, 'datetime') else 'unknown')
        }
    }
    
    # Handle optimizer state
    if preserve_optimizer and 'optimizer_state_dict' in v0_checkpoint:
        print(f"\n⚠️ OPTIMIZER STATE HANDLING:")
        print(f"   V0 optimizer state will be DISCARDED due to dimension mismatch")
        print(f"   V1 training will create fresh optimizer state")
    
    # Create minimal optimizer state for RSL-RL compatibility  
    v1_checkpoint['optimizer_state_dict'] = {
        'state': {},
        'param_groups': [
            {
                'lr': 0.0001,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 0,
                'amsgrad': False,
                'maximize': False,
                'foreach': None,
                'capturable': False,
                'differentiable': False,
                'fused': None,
                'params': []
            }
        ]
    }
    
    # Save V1 model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(v1_checkpoint, output_path)
    
    print(f"\n✅ TRANSFER LEARNING COMPLETE!")
    print(f"   📁 V1 Model saved: {output_path}")
    print(f"   🎯 Input expansion: 28D → 38D (+10 dimensions)")
    print(f"   🧠 Strategy: {expansion_strategy}")
    print(f"   🔄 Iterations preserved: {v1_checkpoint['iter']}")
    
    # Verification
    verification_checkpoint = torch.load(output_path, map_location='cpu')
    actor_weight = verification_checkpoint['model_state_dict']['actor.0.weight']
    critic_weight = verification_checkpoint['model_state_dict']['critic.0.weight']
    
    print(f"\n🔍 VERIFICATION:")
    print(f"   🎭 Actor input dimensions: {actor_weight.shape[1]}")
    print(f"   🔍 Critic input dimensions: {critic_weight.shape[1]}")
    
    if actor_weight.shape[1] == 38 and critic_weight.shape[1] == 38:
        print(f"   ✅ Transfer learning successful!")
    else:
        print(f"   ❌ Transfer learning failed!")
    
    return output_path

if __name__ == "__main__":
    # Define paths
    V0_MODEL_PATH = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/2025-08-19_13-19-01/model_335597.pt"
    V1_OUTPUT_DIR = "/home/nvidia/muammerrepolar/isaac_so_arm101/logs/rsl_rl/so_arm100_picknplace_v1/professional_transfer_v1"
    V1_MODEL_PATH = f"{V1_OUTPUT_DIR}/model_335597_v1.pt"
    
    # Perform professional transfer learning
    result_path = create_v0_to_v1_transfer_model(
        v0_model_path=V0_MODEL_PATH,
        output_path=V1_MODEL_PATH,
        expansion_strategy='orthogonal',  # Best for RL according to research
        preserve_optimizer=False
    )
    
    print(f"\n🚀 READY FOR V1 TRAINING!")
    print(f"Use this command:")
    print(f"python scripts/rsl_rl/train.py --task=SO-ARM100-Lift-Cube-Picknplace-v1 --checkpoint=professional_transfer_v1 --resume --num_envs=64 --max_iterations=1000 --headless --logger=tensorboard")
    
    print(f"\n📊 TRAINING EXPECTATIONS:")
    print(f"   🎯 V0 knowledge preserved for first 28 dimensions")
    print(f"   🆕 New 10 dimensions will learn cube2 and stacking features")
    print(f"   📈 Training should converge faster than fresh start")
    print(f"   🔄 Monitor first few iterations for stability")