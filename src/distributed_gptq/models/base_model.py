"""
Base model interface for distributed GPTQ quantization.

This module provides the abstract base class that all quantizable models
should inherit from.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union


class QuantizableModel(ABC):
    """
    Abstract base class for models that can be quantized using GPTQ.
    """
    
    @abstractmethod
    def get_layers(self) -> Dict[str, nn.Module]:
        """
        Get all layers that should be quantized.
        
        Returns:
            Dictionary mapping layer names to modules
        """
        pass
    
    @abstractmethod
    def get_layer_dependencies(self) -> Dict[str, List[str]]:
        """
        Get dependencies between layers for proper quantization order.
        
        Returns:
            Dictionary mapping layer names to their dependencies
        """
        pass
    
    @abstractmethod
    def prepare_calibration_data(self, calibration_dataset) -> torch.utils.data.DataLoader:
        """
        Prepare calibration data for quantization.
        
        Args:
            calibration_dataset: Raw calibration dataset
            
        Returns:
            DataLoader for calibration
        """
        pass
    
    @abstractmethod
    def forward_to_layer(self, inputs: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Forward pass up to a specific layer.
        
        Args:
            inputs: Input tensor
            layer_name: Target layer name
            
        Returns:
            Output tensor at the target layer
        """
        pass
    
    @abstractmethod
    def replace_layer(self, layer_name: str, new_layer: nn.Module):
        """
        Replace a layer with a quantized version.
        
        Args:
            layer_name: Name of layer to replace
            new_layer: New quantized layer
        """
        pass


class BaseModelWrapper(QuantizableModel):
    """
    Base wrapper for PyTorch models to make them quantizable.
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model wrapper.
        
        Args:
            model: PyTorch model to wrap
            config: Configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self._layer_cache = {}
        self._dependency_cache = None
    
    def get_layers(self) -> Dict[str, nn.Module]:
        """
        Get all Linear layers in the model.
        
        Returns:
            Dictionary of layer names to modules
        """
        if not self._layer_cache:
            self._layer_cache = {}
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    self._layer_cache[name] = module
        
        return self._layer_cache
    
    def get_layer_dependencies(self) -> Dict[str, List[str]]:
        """
        Analyze model structure to determine layer dependencies.
        
        Returns:
            Dictionary of dependencies
        """
        if self._dependency_cache is None:
            # Simple implementation: assume sequential dependencies
            layers = list(self.get_layers().keys())
            dependencies = {}
            
            for i, layer_name in enumerate(layers):
                if i == 0:
                    dependencies[layer_name] = []
                else:
                    dependencies[layer_name] = [layers[i-1]]
            
            self._dependency_cache = dependencies
        
        return self._dependency_cache
    
    def prepare_calibration_data(self, calibration_dataset) -> torch.utils.data.DataLoader:
        """
        Prepare calibration data.
        
        Args:
            calibration_dataset: Raw calibration dataset
            
        Returns:
            DataLoader for calibration
        """
        batch_size = self.config.get('calibration_batch_size', 1)
        
        if hasattr(calibration_dataset, '__iter__'):
            # Already a DataLoader or iterable
            return calibration_dataset
        
        return torch.utils.data.DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def forward_to_layer(self, inputs: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Forward pass up to a specific layer.
        
        Args:
            inputs: Input tensor
            layer_name: Target layer name
            
        Returns:
            Output tensor at the target layer
        """
        # This is a simplified implementation
        # In practice, you'd need to trace the model execution
        with torch.no_grad():
            x = inputs
            
            for name, module in self.model.named_modules():
                if name == layer_name:
                    break
                
                # Apply module if it's in the forward path
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.LayerNorm)):
                    x = module(x)
            
            return x
    
    def replace_layer(self, layer_name: str, new_layer: nn.Module):
        """
        Replace a layer with a quantized version.
        
        Args:
            layer_name: Name of layer to replace
            new_layer: New quantized layer
        """
        # Navigate to the parent module and replace the layer
        parts = layer_name.split('.')
        parent = self.model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_layer)
        
        # Update cache
        if layer_name in self._layer_cache:
            self._layer_cache[layer_name] = new_layer
    
    def get_model(self) -> nn.Module:
        """
        Get the wrapped model.
        
        Returns:
            The PyTorch model
        """
        return self.model
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode."""
        self.model.train(mode)
        return self
    
    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        self.model = self.model.to(device)
        return self
    
    def state_dict(self) -> Dict[str, Any]:
        """Get model state dict."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state dict."""
        self.model.load_state_dict(state_dict)
    
    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()
    
    def named_parameters(self):
        """Get named model parameters."""
        return self.model.named_parameters()
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)


class ModelInfo:
    """
    Container for model metadata and statistics.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize model info.
        
        Args:
            model: PyTorch model
        """
        self.model = model
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute model statistics."""
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Count different layer types
        self.layer_counts = {}
        for module in self.model.modules():
            layer_type = type(module).__name__
            self.layer_counts[layer_type] = self.layer_counts.get(layer_type, 0) + 1
        
        # Estimate memory usage (rough)
        self.memory_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def summary(self) -> Dict[str, Any]:
        """
        Get model summary.
        
        Returns:
            Dictionary with model statistics
        """
        return {
            'total_parameters': self.total_params,
            'trainable_parameters': self.trainable_params,
            'layer_counts': self.layer_counts,
            'memory_mb': self.memory_mb
        }
    
    def __str__(self) -> str:
        """String representation of model info."""
        return f"ModelInfo(params={self.total_params:,}, memory={self.memory_mb:.1f}MB)"
