"""
Quantized layer implementations for GPTQ.

This module provides quantized versions of standard PyTorch layers
that are compatible with the GPTQ quantization scheme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer implementing GPTQ quantization.
    """
    
    def __init__(
        self,
        bits: int,
        group_size: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        """
        Initialize quantized linear layer.
        
        Args:
            bits: Number of bits for quantization (typically 4 or 8)
            group_size: Size of quantization groups
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias
        """
        super().__init__()
        
        self.bits = bits
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        
        # Calculate number of groups
        self.num_groups = math.ceil(in_features / group_size)
        
        # Register quantized weights and scales
        if bits == 4:
            # For 4-bit, pack weights into uint8 (2 weights per byte)
            weight_shape = (out_features, math.ceil(in_features / 2))
        else:
            weight_shape = (out_features, in_features)
        
        self.register_buffer('qweight', torch.zeros(weight_shape, dtype=torch.uint8))
        self.register_buffer('scales', torch.zeros((out_features, self.num_groups), dtype=torch.float16))
        self.register_buffer('zeros', torch.zeros((out_features, self.num_groups), dtype=torch.float16))
        
        # Optional group indices for advanced quantization schemes
        self.register_buffer('g_idx', torch.tensor(range(in_features), dtype=torch.int32))
        
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features)))
        else:
            self.register_parameter('bias', None)
    
    def pack_weights(self, weight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
        """
        Pack weights into quantized format.
        
        Args:
            weight: Quantized weight tensor (uint8)
            scales: Scale factors
            zeros: Zero points
        """
        if self.bits == 4:
            # Pack 4-bit weights
            packed_weight = torch.zeros(
                (weight.shape[0], math.ceil(weight.shape[1] / 2)),
                dtype=torch.uint8,
                device=weight.device
            )
            
            # Pack pairs of 4-bit values into single bytes
            for i in range(0, weight.shape[1], 2):
                if i + 1 < weight.shape[1]:
                    packed_weight[:, i // 2] = weight[:, i] | (weight[:, i + 1] << 4)
                else:
                    packed_weight[:, i // 2] = weight[:, i]
            
            self.qweight.data = packed_weight
        else:
            self.qweight.data = weight
        
        self.scales.data = scales
        self.zeros.data = zeros
    
    def unpack_weights(self) -> torch.Tensor:
        """
        Unpack quantized weights to full precision.
        
        Returns:
            Unpacked weight tensor
        """
        if self.bits == 4:
            # Unpack 4-bit weights
            weight = torch.zeros(
                (self.qweight.shape[0], self.in_features),
                dtype=torch.uint8,
                device=self.qweight.device
            )
            
            for i in range(0, self.in_features, 2):
                if i + 1 < self.in_features:
                    weight[:, i] = self.qweight[:, i // 2] & 0xF
                    weight[:, i + 1] = (self.qweight[:, i // 2] >> 4) & 0xF
                else:
                    weight[:, i] = self.qweight[:, i // 2] & 0xF
        else:
            weight = self.qweight
        
        # Dequantize: weight = (quantized - zero) * scale
        weight = weight.float()
        
        # Apply group-wise dequantization
        dequant_weight = torch.zeros_like(weight, dtype=torch.float32)
        
        for group_idx in range(self.num_groups):
            start_idx = group_idx * self.group_size
            end_idx = min(start_idx + self.group_size, self.in_features)
            
            group_weight = weight[:, start_idx:end_idx]
            group_scales = self.scales[:, group_idx:group_idx+1]
            group_zeros = self.zeros[:, group_idx:group_idx+1]
            
            dequant_weight[:, start_idx:end_idx] = (group_weight - group_zeros) * group_scales
        
        return dequant_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantized linear layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Fast path: use custom kernels if available
        if hasattr(torch.ops, 'gptq'):
            # Use optimized GPTQ kernel
            return torch.ops.gptq.linear(x, self.qweight, self.scales, self.zeros, self.g_idx, self.bias)
        
        # Fallback: dequantize and use standard linear
        weight = self.unpack_weights()
        return F.linear(x, weight, self.bias)
    
    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128
    ) -> 'QuantizedLinear':
        """
        Create a quantized linear layer from a standard linear layer.
        
        Args:
            linear_layer: Original linear layer
            bits: Number of bits for quantization
            group_size: Size of quantization groups
            
        Returns:
            Quantized linear layer
        """
        quantized = cls(
            bits=bits,
            group_size=group_size,
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None
        )
        
        if linear_layer.bias is not None:
            quantized.bias.data = linear_layer.bias.data
        
        return quantized


class QuantizedEmbedding(nn.Module):
    """
    Quantized embedding layer.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: int = 8,
        padding_idx: Optional[int] = None
    ):
        """
        Initialize quantized embedding.
        
        Args:
            num_embeddings: Number of embeddings
            embedding_dim: Embedding dimension
            bits: Number of bits for quantization
            padding_idx: Padding index
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bits = bits
        self.padding_idx = padding_idx
        
        # For embeddings, we typically use 8-bit quantization
        self.register_buffer('qweight', torch.zeros((num_embeddings, embedding_dim), dtype=torch.uint8))
        self.register_buffer('scales', torch.zeros(num_embeddings, dtype=torch.float32))
        self.register_buffer('zeros', torch.zeros(num_embeddings, dtype=torch.float32))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantized embedding.
        
        Args:
            input: Input indices
            
        Returns:
            Embedded tensor
        """
        # Dequantize embeddings
        weight = (self.qweight.float() - self.zeros.unsqueeze(1)) * self.scales.unsqueeze(1)
        return F.embedding(input, weight, self.padding_idx)


class QuantizedLayerNorm(nn.Module):
    """
    Quantized Layer Normalization.
    Note: LayerNorm is typically not quantized in GPTQ, but included for completeness.
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        """
        Initialize quantized layer norm.
        
        Args:
            normalized_shape: Shape for normalization
            eps: Epsilon for numerical stability
            elementwise_affine: Whether to use learnable affine parameters
        """
        super().__init__()
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through layer norm.
        
        Args:
            input: Input tensor
            
        Returns:
            Normalized tensor
        """
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


def replace_with_quantized_layers(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    layer_types: Optional[list] = None
) -> nn.Module:
    """
    Replace standard layers with quantized versions.
    
    Args:
        model: Model to quantize
        bits: Number of bits for quantization
        group_size: Size of quantization groups
        layer_types: Types of layers to replace (default: [nn.Linear])
        
    Returns:
        Model with quantized layers
    """
    if layer_types is None:
        layer_types = [nn.Linear]
    
    for name, module in model.named_children():
        if type(module) in layer_types:
            if isinstance(module, nn.Linear):
                quantized_layer = QuantizedLinear.from_linear(module, bits, group_size)
                setattr(model, name, quantized_layer)
        elif len(list(module.children())) > 0:
            # Recursively replace in child modules
            replace_with_quantized_layers(module, bits, group_size, layer_types)
    
    return model
