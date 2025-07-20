"""
Dequantization logic for GPTQ quantized models.

This module provides functionality to dequantize models that have been
quantized using the GPTQ algorithm.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import numpy as np


class GPTQDequantizer:
    """
    Handles dequantization of GPTQ quantized models.
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        """
        Initialize the dequantizer.
        
        Args:
            bits: Number of bits used for quantization
            group_size: Size of quantization groups
        """
        self.bits = bits
        self.group_size = group_size
        self.max_val = 2 ** bits - 1
    
    def dequantize_weight(
        self, 
        qweight: torch.Tensor, 
        scales: torch.Tensor, 
        zeros: torch.Tensor,
        g_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Dequantize a quantized weight tensor.
        
        Args:
            qweight: Quantized weight tensor
            scales: Scale factors for dequantization
            zeros: Zero points for dequantization
            g_idx: Group indices for advanced quantization schemes
            
        Returns:
            Dequantized weight tensor
        """
        # Unpack quantized weights if needed
        if self.bits == 4:
            # For 4-bit quantization, weights are packed
            weight = self._unpack_4bit(qweight)
        else:
            weight = qweight.float()
        
        # Apply dequantization formula
        if g_idx is not None:
            # Advanced grouping
            scales = scales[g_idx]
            zeros = zeros[g_idx] if zeros.numel() > 1 else zeros
        
        # Dequantize: weight = (quantized - zero) * scale
        weight = (weight - zeros) * scales
        
        return weight
    
    def _unpack_4bit(self, qweight: torch.Tensor) -> torch.Tensor:
        """
        Unpack 4-bit quantized weights.
        
        Args:
            qweight: Packed 4-bit weights
            
        Returns:
            Unpacked weights
        """
        # Each byte contains two 4-bit values
        weight = torch.zeros(
            (qweight.shape[0], qweight.shape[1] * 2),
            dtype=torch.uint8,
            device=qweight.device
        )
        
        # Extract lower and upper 4 bits
        weight[:, ::2] = qweight & 0xF  # Lower 4 bits
        weight[:, 1::2] = (qweight >> 4) & 0xF  # Upper 4 bits
        
        return weight.float()
    
    def dequantize_layer(self, layer: nn.Module) -> nn.Module:
        """
        Dequantize a quantized layer.
        
        Args:
            layer: Quantized layer
            
        Returns:
            Dequantized layer
        """
        if hasattr(layer, 'qweight'):
            # This is a quantized layer
            weight = self.dequantize_weight(
                layer.qweight,
                layer.scales,
                layer.zeros,
                getattr(layer, 'g_idx', None)
            )
            
            # Create new linear layer
            new_layer = nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=hasattr(layer, 'bias') and layer.bias is not None
            )
            
            new_layer.weight.data = weight
            if hasattr(layer, 'bias') and layer.bias is not None:
                new_layer.bias.data = layer.bias.data
                
            return new_layer
        
        return layer
    
    def dequantize_model(self, model: nn.Module) -> nn.Module:
        """
        Dequantize an entire model.
        
        Args:
            model: Quantized model
            
        Returns:
            Dequantized model
        """
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                # Recursively dequantize child modules
                setattr(model, name, self.dequantize_model(module))
            else:
                # Dequantize leaf modules
                setattr(model, name, self.dequantize_layer(module))
        
        return model


def dequantize_model(
    model: nn.Module, 
    bits: int = 4, 
    group_size: int = 128
) -> nn.Module:
    """
    Convenience function to dequantize a model.
    
    Args:
        model: Quantized model
        bits: Number of bits used for quantization
        group_size: Size of quantization groups
        
    Returns:
        Dequantized model
    """
    dequantizer = GPTQDequantizer(bits=bits, group_size=group_size)
    return dequantizer.dequantize_model(model)
