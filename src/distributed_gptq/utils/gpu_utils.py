"""
GPU memory management utilities for distributed GPTQ.

This module provides utilities for managing GPU memory efficiently
during the quantization process.
"""

import torch
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Union
import logging
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """
    Manages GPU memory allocation and cleanup.
    """
    
    def __init__(self, device: Union[str, torch.device] = None):
        """
        Initialize GPU memory manager.
        
        Args:
            device: Target device (if None, uses current device)
        """
        if device is None:
            self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.peak_memory = 0
        self.memory_snapshots = []
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory information.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        free = total - reserved
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total
        }
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def snapshot_memory(self, label: str = ""):
        """
        Take a memory snapshot.
        
        Args:
            label: Optional label for the snapshot
        """
        info = self.get_memory_info()
        info['label'] = label
        self.memory_snapshots.append(info)
        
        current_allocated = info['allocated']
        if current_allocated > self.peak_memory:
            self.peak_memory = current_allocated
        
        logger.debug(f"Memory snapshot '{label}': {current_allocated:.2f}GB allocated")


def get_gpu_memory_info(device: Optional[Union[str, torch.device]] = None) -> Dict[str, float]:
    """Return GPU memory statistics for the given device."""
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    if not torch.cuda.is_available() or (isinstance(device, str) and device == "cpu"):
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}

    dev = torch.device(device)
    allocated = torch.cuda.memory_allocated(dev) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(dev) / 1024 ** 3
    total = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3
    free = total - reserved

    return {"allocated": allocated, "reserved": reserved, "free": free, "total": total}


def clear_gpu_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    """Clear the CUDA memory cache."""
    if not torch.cuda.is_available():
        return
    if device is not None:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()


def get_device_capability(device: Optional[Union[int, torch.device]] = None) -> Tuple[int, int]:
    """Return compute capability of the given CUDA device."""
    if not torch.cuda.is_available():
        return (0, 0)
    dev = device if device is not None else torch.cuda.current_device()
    dev = dev.index if isinstance(dev, torch.device) else dev
    return torch.cuda.get_device_capability(dev)

def get_free_gpu_memory():
    """Get free GPU memory."""
    import torch
    return torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


def set_gpu_memory_limit(limit):
    """Set GPU memory limit."""
    import torch
    if limit > 0:
        torch.cuda.set_per_process_memory_fraction(limit)


def check_gpu_availability():
    """Check if GPU is available."""
    import torch
    return torch.cuda.is_available()


def get_gpu_count():
    """Get number of available GPUs."""
    import torch
    return torch.cuda.device_count()


def get_current_device():
    """Get current GPU device."""
    import torch
    return torch.cuda.current_device()


def estimate_model_memory(
    num_parameters: int,
    dtype: torch.dtype = torch.float16,
    overhead_factor: float = 1.2
) -> float:
    """
    Estimate memory requirements for a model.
    
    Args:
        num_parameters: Number of model parameters
        dtype: Parameter data type
        overhead_factor: Memory overhead factor
        
    Returns:
        Estimated memory in GB
    """
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    base_memory = num_parameters * bytes_per_param / 1024**3
    return base_memory * overhead_factor


def get_optimal_batch_size(
    available_memory_gb: float,
    model_memory_gb: float,
    sequence_length: int = 512,
    safety_factor: float = 0.8
) -> int:
    """
    Calculate optimal batch size based on available memory.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        model_memory_gb: Model memory footprint in GB
        sequence_length: Input sequence length
        safety_factor: Safety factor for memory allocation
        
    Returns:
        Recommended batch size
    """
    # Reserve memory for model
    working_memory = (available_memory_gb - model_memory_gb) * safety_factor
    
    if working_memory <= 0:
        return 1
    
    # Estimate memory per sample (rough approximation)
    # Assumes float16 activations
    memory_per_sample = sequence_length * 2 / 1024**3  # Very rough estimate
    
    batch_size = max(1, int(working_memory / memory_per_sample))
    return min(batch_size, 32)  # Cap at reasonable maximum


@contextmanager
def gpu_memory_guard(device: Optional[Union[str, torch.device]] = None):
    """
    Context manager that clears GPU memory on exit.
    
    Args:
        device: Target device
    """
    try:
        yield
    finally:
        if torch.cuda.is_available():
            if device is not None:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
        gc.collect()

def get_device_name(device_id):
    import torch
    return torch.cuda.get_device_name(device_id)