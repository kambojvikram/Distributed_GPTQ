"""
Core GPTQ quantization algorithm implementation.
Based on "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
import logging
from tqdm import tqdm
import math

logger = logging.getLogger(__name__)


class GPTQ:
    """
    GPTQ quantization algorithm implementation.
    
    This class implements the GPTQ algorithm for quantizing neural network layers
    using the Optimal Brain Surgeon (OBS) framework with lazy batch updates.
    """
    
    def __init__(
        self,
        layer: nn.Module,
        bits: int = 4,
        group_size: int = 128,
        actorder: bool = False,
        percdamp: float = 0.01,
        blocksize: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize GPTQ quantizer.
        
        Args:
            layer: The layer to quantize
            bits: Number of bits for quantization (2, 3, 4, 8)
            group_size: Size of quantization groups
            actorder: Whether to use activation order
            percdamp: Percentage of average Hessian diagonal to add for dampening
            blocksize: Block size for quantization
            device: Device to use for computation
        """
        self.layer = layer
        self.bits = bits
        self.group_size = group_size
        self.actorder = actorder
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Quantization parameters
        self.maxq = 2 ** self.bits - 1
        self.scale = None
        self.zero = None
        
        # Hessian and error tracking
        self.H = None
        self.nsamples = 0
        
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Add a batch of data to update Hessian matrix.
        
        Args:
            inp: Input activations
            out: Output gradients
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        batch_size = inp.shape[0]
        
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        self.nsamples += batch_size
        
        # Initialize Hessian
        if self.H is None:
            self.H = torch.zeros((inp.shape[0], inp.shape[0]), device=self.device)
        
        # Update Hessian: H = X @ X^T
        inp = inp.float()
        self.H += inp @ inp.t()
        
    def quantize(self) -> Dict[str, Any]:
        """
        Perform GPTQ quantization on the layer.
        
        Returns:
            Dictionary containing quantization results and statistics
        """
        logger.info(f"Starting GPTQ quantization with {self.bits} bits")
        
        # Get weight matrix
        W = self.layer.weight.data.clone()
        W = W.float()
        
        # Handle different layer types
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, nn.Linear):
            W = W.t()
        
        rows, columns = W.shape
        
        # Normalize Hessian
        self.H /= self.nsamples
        
        # Add dampening
        diag_H = torch.diag(self.H)
        diag_mean = diag_H.mean()
        diag_H += self.percdamp * diag_mean
        torch.diagonal(self.H).copy_(diag_H)
        
        # Prepare quantization
        self.H_inv = torch.cholesky_inverse(torch.linalg.cholesky(self.H))
        
        # Initialize quantization parameters
        self.scale = torch.zeros((rows, columns // self.group_size), device=self.device)
        self.zero = torch.zeros((rows, columns // self.group_size), device=self.device)
        
        # Quantization error
        total_error = 0.0
        
        # Quantize in blocks
        for i1 in tqdm(range(0, columns, self.blocksize), desc="Quantizing blocks"):
            i2 = min(i1 + self.blocksize, columns)
            block_columns = i2 - i1
            
            # Extract block
            W_block = W[:, i1:i2].clone()
            Q_block = torch.zeros_like(W_block)
            
            # Get block Hessian
            H_block = self.H[i1:i2, i1:i2]
            H_inv_block = self.H_inv[i1:i2, i1:i2]
            
            # Initialize error
            Err = torch.zeros_like(W_block)
            Losses = torch.zeros_like(W_block)
            
            # Quantize columns
            for j in range(block_columns):
                w = W_block[:, j]
                
                # Get diagonal element
                d = H_inv_block[j, j]
                
                # Calculate group index
                group_idx = (i1 + j) // self.group_size
                
                # Find optimal scale and zero point for this group
                if (i1 + j) % self.group_size == 0:
                    g_start = i1 + j
                    g_end = min(g_start + self.group_size, columns)
                    w_group = W[:, g_start:g_end]
                    
                    # Calculate scale and zero point
                    w_max = w_group.max(dim=1, keepdim=True)[0]
                    w_min = w_group.min(dim=1, keepdim=True)[0]
                    
                    self.scale[:, group_idx] = (w_max - w_min).squeeze() / self.maxq
                    self.zero[:, group_idx] = torch.round(-w_min.squeeze() / self.scale[:, group_idx])
                
                # Quantize
                scale = self.scale[:, group_idx].unsqueeze(1)
                zero = self.zero[:, group_idx].unsqueeze(1)
                
                q = torch.clamp(torch.round(w / scale + zero), 0, self.maxq)
                Q_block[:, j] = q.squeeze()
                
                # Dequantize
                w_hat = (q - zero) * scale
                w_hat = w_hat.squeeze()
                
                # Calculate error
                err = (w - w_hat) / d
                
                # Update remaining weights
                if j < block_columns - 1:
                    W_block[:, j+1:] -= err.unsqueeze(1) @ H_inv_block[j, j+1:].unsqueeze(0)
                
                # Track error
                Err[:, j] = err
                total_error += (err ** 2).sum().item()
            
            # Store quantized block
            W[:, i1:i2] = Q_block
        
        # Update layer weights with quantized values
        if isinstance(self.layer, nn.Linear):
            self.layer.weight.data = W.t()
        else:
            self.layer.weight.data = W.reshape(self.layer.weight.shape)
        
        # Calculate statistics
        stats = {
            'bits': self.bits,
            'group_size': self.group_size,
            'total_error': total_error,
            'compression_ratio': 32.0 / self.bits,
            'scale_shape': self.scale.shape,
            'zero_shape': self.zero.shape,
        }
        
        logger.info(f"Quantization complete. Total error: {total_error:.4f}")
        
        return stats
    
    def pack_weights(self) -> Dict[str, torch.Tensor]:
        """
        Pack quantized weights for efficient storage.
        
        Returns:
            Dictionary containing packed weights and quantization parameters
        """
        W = self.layer.weight.data
        
        if isinstance(self.layer, nn.Linear):
            W = W.t()
        
        # Pack weights according to bit width
        if self.bits == 4:
            # Pack two 4-bit values into one 8-bit value
            W_packed = torch.zeros(
                (W.shape[0], W.shape[1] // 2), 
                dtype=torch.uint8, 
                device=W.device
            )
            
            for i in range(0, W.shape[1], 2):
                W_packed[:, i // 2] = (W[:, i] * 16 + W[:, i + 1]).to(torch.uint8)
        
        elif self.bits == 2:
            # Pack four 2-bit values into one 8-bit value
            W_packed = torch.zeros(
                (W.shape[0], W.shape[1] // 4), 
                dtype=torch.uint8, 
                device=W.device
            )
            
            for i in range(0, W.shape[1], 4):
                W_packed[:, i // 4] = (
                    W[:, i] * 64 + 
                    W[:, i + 1] * 16 + 
                    W[:, i + 2] * 4 + 
                    W[:, i + 3]
                ).to(torch.uint8)
        
        else:
            W_packed = W.to(torch.int8 if self.bits == 8 else torch.uint8)
        
        return {
            'packed_weights': W_packed,
            'scale': self.scale,
            'zero': self.zero,
            'bits': self.bits,
            'group_size': self.group_size,
            'shape': W.shape,
        }


class GPTQMultiGPU:
    """
    Multi-GPU GPTQ quantization with distributed processing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        group_size: int = 128,
        actorder: bool = False,
        percdamp: float = 0.01,
        blocksize: int = 128,
        devices: Optional[List[torch.device]] = None,
    ):
        """
        Initialize multi-GPU GPTQ quantizer.
        
        Args:
            model: Model to quantize
            bits: Number of bits for quantization
            group_size: Size of quantization groups
            actorder: Whether to use activation order
            percdamp: Percentage of dampening
            blocksize: Block size for quantization
            devices: List of devices to use
        """
        self.model = model
        self.bits = bits
        self.group_size = group_size
        self.actorder = actorder
        self.percdamp = percdamp
        self.blocksize = blocksize
        
        # Setup devices
        if devices is None:
            self.devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        else:
            self.devices = devices
        
        self.layer_quantizers = {}
        
    def prepare_layer(self, layer_name: str, layer: nn.Module) -> GPTQ:
        """
        Prepare a layer for quantization.
        
        Args:
            layer_name: Name of the layer
            layer: Layer module
            
        Returns:
            GPTQ quantizer for the layer
        """
        # Distribute layers across devices
        device_idx = hash(layer_name) % len(self.devices)
        device = self.devices[device_idx]
        
        # Move layer to device
        layer = layer.to(device)
        
        # Create quantizer
        quantizer = GPTQ(
            layer=layer,
            bits=self.bits,
            group_size=self.group_size,
            actorder=self.actorder,
            percdamp=self.percdamp,
            blocksize=self.blocksize,
            device=device
        )
        
        self.layer_quantizers[layer_name] = quantizer
        return quantizer
    
    def add_batch_to_all(self, inputs: Dict[str, torch.Tensor]):
        """
        Add batch to all layer quantizers.
        
        Args:
            inputs: Dictionary of layer inputs
        """
        for layer_name, quantizer in self.layer_quantizers.items():
            if layer_name in inputs:
                inp = inputs[layer_name]
                # For now, use dummy output gradients
                out = torch.ones_like(inp)
                quantizer.add_batch(inp, out)
    
    def quantize_all_layers(self) -> Dict[str, Dict[str, Any]]:
        """
        Quantize all prepared layers.
        
        Returns:
            Dictionary of quantization statistics for each layer
        """
        results = {}
        
        for layer_name, quantizer in tqdm(self.layer_quantizers.items(), desc="Quantizing layers"):
            logger.info(f"Quantizing layer: {layer_name}")
            stats = quantizer.quantize()
            results[layer_name] = stats
        
        return results
    
    def get_quantized_model(self) -> nn.Module:
        """
        Get the quantized model.
        
        Returns:
            Quantized model
        """
        return self.model


def quantize_layer_gptq(
    layer: nn.Module,
    calibration_data: torch.utils.data.DataLoader,
    bits: int = 4,
    group_size: int = 128,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to quantize a single layer using GPTQ.
    
    Args:
        layer: Layer to quantize
        calibration_data: Calibration dataset
        bits: Number of bits for quantization
        group_size: Size of quantization groups
        **kwargs: Additional arguments for GPTQ
        
    Returns:
        Quantization statistics
    """
    quantizer = GPTQ(layer, bits=bits, group_size=group_size, **kwargs)
    
    # Add calibration data
    for batch in calibration_data:
        if isinstance(batch, (tuple, list)):
            inp, out = batch
        else:
            inp = batch
            out = torch.ones_like(inp)
        
        quantizer.add_batch(inp, out)
    
    # Quantize
    stats = quantizer.quantize()
    return stats