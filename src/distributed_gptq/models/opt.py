"""
OPT model support for GPTQ quantization.

This module provides specific implementations for quantizing OPT models
from Meta AI (Open Pre-trained Transformer).
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import re

from .base_model import QuantizableModel
from .layers import QuantizedLinear


class OPTQuantizableModel(QuantizableModel):
    """
    Quantizable wrapper for OPT models.
    
    Handles the specific architecture and layer naming conventions
    of OPT models for efficient quantization.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize OPT quantizable model.
        
        Args:
            model: Original OPT model from transformers
        """
        super().__init__()
        self.model = model
        self.config = getattr(model, 'config', None)
        
        # OPT-specific layer patterns
        self.linear_layer_patterns = [
            r'model\.decoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)',
            r'model\.decoder\.layers\.\d+\.fc[12]',
            r'lm_head'
        ]
        
        # Store original forward method
        self._original_forward = model.forward
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def get_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Get all quantizable layers in the OPT model.
        
        Returns:
            List of (layer_name, layer_module) tuples
        """
        quantizable_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this layer should be quantized
                if self._should_quantize_layer(name, module):
                    quantizable_layers.append((name, module))
        
        return quantizable_layers
    
    def _should_quantize_layer(self, name: str, module: nn.Module) -> bool:
        """
        Determine if a layer should be quantized.
        
        Args:
            name: Layer name
            module: Layer module
            
        Returns:
            True if layer should be quantized
        """
        # Skip very small layers
        if hasattr(module, 'weight') and module.weight.numel() < 1000:
            return False
        
        # Skip embedding layers (they should be handled separately)
        if 'embed' in name.lower() and 'pos_embed' not in name.lower():
            return False
        
        # Check against patterns
        for pattern in self.linear_layer_patterns:
            if re.search(pattern, name):
                return True
        
        return False
    
    def get_layer_dependencies(self) -> Dict[str, List[str]]:
        """
        Get dependencies between layers for quantization ordering.
        
        Returns:
            Dictionary mapping layer names to their dependencies
        """
        dependencies = {}
        
        # For OPT, attention layers can be quantized independently
        # FC layers should be done after attention in the same block
        for name, _ in self.get_layers():
            dependencies[name] = []
            
            # Extract layer number if present
            layer_match = re.search(r'layers\.(\d+)\.', name)
            if layer_match:
                layer_num = layer_match.group(1)
                
                # FC layers depend on attention layers in the same block
                if 'fc' in name:
                    attn_layers = [
                        f'model.decoder.layers.{layer_num}.self_attn.q_proj',
                        f'model.decoder.layers.{layer_num}.self_attn.k_proj',
                        f'model.decoder.layers.{layer_num}.self_attn.v_proj',
                        f'model.decoder.layers.{layer_num}.self_attn.out_proj'
                    ]
                    dependencies[name].extend(attn_layers)
        
        return dependencies
    
    def replace_layer(self, layer_name: str, new_layer: nn.Module):
        """
        Replace a layer with its quantized version.
        
        Args:
            layer_name: Name of the layer to replace
            new_layer: New quantized layer
        """
        # Navigate to the parent module
        parts = layer_name.split('.')
        parent = self.model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the layer
        setattr(parent, parts[-1], new_layer)
    
    def get_attention_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Get all attention projection layers.
        
        Returns:
            List of attention layer (name, module) tuples
        """
        attention_layers = []
        
        for name, module in self.get_layers():
            if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
                attention_layers.append((name, module))
        
        return attention_layers
    
    def get_mlp_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Get all MLP/FC layers.
        
        Returns:
            List of MLP layer (name, module) tuples
        """
        mlp_layers = []
        
        for name, module in self.get_layers():
            if 'fc1' in name or 'fc2' in name:
                mlp_layers.append((name, module))
        
        return mlp_layers
    
    def get_output_layer(self) -> Optional[Tuple[str, nn.Module]]:
        """
        Get the language modeling head layer.
        
        Returns:
            (name, module) tuple for LM head or None if not found
        """
        for name, module in self.model.named_modules():
            if name == 'lm_head' and isinstance(module, nn.Linear):
                return (name, module)
        
        return None
    
    def prepare_for_quantization(self):
        """
        Prepare the model for quantization.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Ensure no gradients are computed
        for param in self.model.parameters():
            param.requires_grad = False
        
        # OPT-specific preparations
        if hasattr(self.model, 'config'):
            # Store original config for reference
            self._original_config = self.model.config
    
    def finalize_quantization(self):
        """
        Finalize the model after quantization.
        """
        # Clean up any temporary states
        if hasattr(self, '_quantization_cache'):
            delattr(self, '_quantization_cache')
    
    def get_grouped_quantization_order(self) -> List[List[str]]:
        """
        Get the optimal order for quantizing layers in groups.
        
        Returns:
            List of layer name groups that can be quantized together
        """
        groups = []
        
        # Group by transformer layer
        layer_groups = {}
        
        for name, _ in self.get_layers():
            layer_match = re.search(r'layers\.(\d+)\.', name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                if layer_num not in layer_groups:
                    layer_groups[layer_num] = {'attention': [], 'mlp': []}
                
                if 'self_attn' in name:
                    layer_groups[layer_num]['attention'].append(name)
                elif 'fc' in name:
                    layer_groups[layer_num]['mlp'].append(name)
            else:
                # Output layer
                groups.append([name])
        
        # Create groups: attention layers first, then MLP layers
        for layer_num in sorted(layer_groups.keys()):
            if layer_groups[layer_num]['attention']:
                groups.append(layer_groups[layer_num]['attention'])
            if layer_groups[layer_num]['mlp']:
                groups.append(layer_groups[layer_num]['mlp'])
        
        return groups


class OPTQuantizationOptimizer:
    """
    Optimization utilities specific to OPT quantization.
    """
    
    @staticmethod
    def get_optimal_group_sizes(model_config) -> Dict[str, int]:
        """
        Get optimal group sizes for different layer types.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Dictionary mapping layer patterns to optimal group sizes
        """
        # Default group sizes based on hidden dimensions
        hidden_size = getattr(model_config, 'hidden_size', 768)
        ffn_dim = getattr(model_config, 'ffn_dim', hidden_size * 4)
        
        group_sizes = {
            'q_proj': min(128, hidden_size // 16),
            'k_proj': min(128, hidden_size // 16),
            'v_proj': min(128, hidden_size // 16),
            'out_proj': min(128, hidden_size // 16),
            'fc1': min(128, ffn_dim // 32),
            'fc2': min(128, hidden_size // 16),
            'lm_head': min(256, hidden_size // 8)
        }
        
        return group_sizes
    
    @staticmethod
    def get_layer_specific_configs(model_config) -> Dict[str, Dict[str, Any]]:
        """
        Get layer-specific quantization configurations.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Dictionary mapping layer patterns to quantization configs
        """
        configs = {}
        
        # OPT attention layers
        attention_config = {
            'bits': 4,
            'actorder': True,
            'percdamp': 0.01
        }
        
        # OPT FC layers
        mlp_config = {
            'bits': 4,
            'actorder': False,
            'percdamp': 0.02
        }
        
        # Output layer
        output_config = {
            'bits': 8,  # More conservative for output
            'actorder': True,
            'percdamp': 0.005
        }
        
        for pattern in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            configs[pattern] = attention_config
        
        for pattern in ['fc1', 'fc2']:
            configs[pattern] = mlp_config
        
        configs['lm_head'] = output_config
        
        return configs


def create_opt_quantizable_model(model: nn.Module) -> OPTQuantizableModel:
    """
    Create a quantizable wrapper for OPT models.
    
    Args:
        model: Original OPT model
        
    Returns:
        Quantizable model wrapper
    """
    return OPTQuantizableModel(model)


def get_opt_calibration_hook(layer_name: str):
    """
    Get a calibration hook specific to OPT layers.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Hook function for calibration data collection
    """
    def hook_fn(module, input, output):
        """Hook function to collect calibration data."""
        if not hasattr(module, '_calibration_data'):
            module._calibration_data = []
        
        # Store input activations
        if isinstance(input, tuple):
            input_tensor = input[0]
        else:
            input_tensor = input
        
        # Only store a subset to manage memory
        if len(module._calibration_data) < 100:
            module._calibration_data.append(input_tensor.detach().cpu())
    
    return hook_fn
