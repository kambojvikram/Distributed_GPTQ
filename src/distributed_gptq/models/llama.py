"""
LLaMA model support for GPTQ quantization.

This module provides specific implementations for quantizing LLaMA/LLaMA2 models.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import re

from .base_model import QuantizableModel
from .layers import QuantizedLinear


class LLaMAQuantizableModel(QuantizableModel):
    """
    Quantizable wrapper for LLaMA/LLaMA2 models.
    
    Handles the specific architecture and layer naming conventions
    of LLaMA models for efficient quantization.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize LLaMA quantizable model.
        
        Args:
            model: Original LLaMA model from transformers
        """
        super().__init__()
        self.model = model
        self.config = getattr(model, 'config', None)
        
        # LLaMA-specific layer patterns
        self.linear_layer_patterns = [
            r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)',
            r'model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)',
            r'lm_head'
        ]
        
        # Store original forward method
        self._original_forward = model.forward
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def get_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Get all quantizable layers in the LLaMA model.
        
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
        # Skip very small layers (embedding-related)
        if hasattr(module, 'weight') and module.weight.numel() < 1000:
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
        
        # For LLaMA, we can quantize attention layers independently
        # but MLP layers within the same block should be done together
        for name, _ in self.get_layers():
            dependencies[name] = []
            
            # Extract layer number if present
            layer_match = re.search(r'layers\.(\d+)\.', name)
            if layer_match:
                layer_num = layer_match.group(1)
                
                # MLP layers depend on attention layers in the same block
                if 'mlp' in name:
                    attn_layers = [
                        f'model.layers.{layer_num}.self_attn.q_proj',
                        f'model.layers.{layer_num}.self_attn.k_proj', 
                        f'model.layers.{layer_num}.self_attn.v_proj',
                        f'model.layers.{layer_num}.self_attn.o_proj'
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
            if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                attention_layers.append((name, module))
        
        return attention_layers
    
    def get_mlp_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Get all MLP layers.
        
        Returns:
            List of MLP layer (name, module) tuples
        """
        mlp_layers = []
        
        for name, module in self.get_layers():
            if any(proj in name for proj in ['gate_proj', 'up_proj', 'down_proj']):
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
        
        This method can be used to set up any necessary states
        or configurations before quantization begins.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Ensure no gradients are computed
        for param in self.model.parameters():
            param.requires_grad = False
    
    def finalize_quantization(self):
        """
        Finalize the model after quantization.
        
        This method can be used to clean up or optimize
        the model after quantization is complete.
        """
        # Clean up any temporary states
        if hasattr(self, '_quantization_cache'):
            delattr(self, '_quantization_cache')


class LLaMA2QuantizableModel(LLaMAQuantizableModel):
    """
    Specialized quantizable model for LLaMA2.
    
    Inherits from LLaMAQuantizableModel but may have
    specific optimizations for LLaMA2 architecture.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        
        # LLaMA2 might have slightly different patterns
        self.linear_layer_patterns.extend([
            r'model\.layers\.\d+\.self_attn\.rotary_emb',  # If present
        ])
    
    def get_grouped_quantization_order(self) -> List[List[str]]:
        """
        Get the optimal order for quantizing layers in groups.
        
        For LLaMA2, we can quantize attention heads in parallel
        and MLP layers together.
        
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
                elif 'mlp' in name:
                    layer_groups[layer_num]['mlp'].append(name)
            else:
                # Output layer
                groups.append([name])
        
        # Create groups: all attention layers first, then MLP layers
        for layer_num in sorted(layer_groups.keys()):
            if layer_groups[layer_num]['attention']:
                groups.append(layer_groups[layer_num]['attention'])
            if layer_groups[layer_num]['mlp']:
                groups.append(layer_groups[layer_num]['mlp'])
        
        return groups


def create_llama_quantizable_model(model: nn.Module) -> LLaMAQuantizableModel:
    """
    Create a quantizable wrapper for LLaMA models.
    
    Args:
        model: Original LLaMA model
        
    Returns:
        Quantizable model wrapper
    """
    # Detect LLaMA version from config or model structure
    if hasattr(model, 'config'):
        model_type = getattr(model.config, 'model_type', 'llama')
        if 'llama' in model_type.lower():
            if hasattr(model.config, 'vocab_size') and model.config.vocab_size > 32000:
                # Likely LLaMA2 (has larger vocab)
                return LLaMA2QuantizableModel(model)
    
    return LLaMAQuantizableModel(model)


def get_llama_calibration_hook(layer_name: str):
    """
    Get a calibration hook specific to LLaMA layers.
    
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


class LLaMAQuantizationOptimizer:
    """
    Optimization utilities specific to LLaMA quantization.
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
        hidden_size = getattr(model_config, 'hidden_size', 4096)
        intermediate_size = getattr(model_config, 'intermediate_size', 11008)
        
        group_sizes = {
            'q_proj': min(128, hidden_size // 32),
            'k_proj': min(128, hidden_size // 32),
            'v_proj': min(128, hidden_size // 32),
            'o_proj': min(128, hidden_size // 32),
            'gate_proj': min(128, intermediate_size // 64),
            'up_proj': min(128, intermediate_size // 64),
            'down_proj': min(128, hidden_size // 32),
            'lm_head': min(256, hidden_size // 16)
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
        
        # Attention layers can use more aggressive quantization
        attention_config = {
            'bits': 4,
            'actorder': True,
            'percdamp': 0.01
        }
        
        # MLP layers might need more conservative settings
        mlp_config = {
            'bits': 4,
            'actorder': False,
            'percdamp': 0.02
        }
        
        # Output layer should be quantized carefully
        output_config = {
            'bits': 8,  # Use 8-bit for output layer
            'actorder': True,
            'percdamp': 0.005
        }
        
        for pattern in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            configs[pattern] = attention_config
        
        for pattern in ['gate_proj', 'up_proj', 'down_proj']:
            configs[pattern] = mlp_config
        
        configs['lm_head'] = output_config
        
        return configs
