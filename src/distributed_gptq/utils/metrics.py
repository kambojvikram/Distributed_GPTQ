"""
Performance metrics and monitoring for distributed GPTQ.

This module provides utilities for measuring and tracking performance
during the quantization process.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path


@dataclass
class QuantizationMetrics:
    """
    Container for quantization performance metrics.
    """
    
    # Timing metrics
    total_time: float = 0.0
    layer_times: Dict[str, float] = field(default_factory=dict)
    calibration_time: float = 0.0
    quantization_time: float = 0.0
    
    # Memory metrics
    peak_memory_gb: float = 0.0
    memory_saved_gb: float = 0.0
    memory_efficiency: float = 0.0
    
    # Model metrics
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    
    # Quality metrics
    perplexity_original: Optional[float] = None
    perplexity_quantized: Optional[float] = None
    perplexity_degradation: Optional[float] = None
    
    # Throughput metrics
    tokens_per_second: Optional[float] = None
    samples_per_second: Optional[float] = None
    
    # Error metrics
    quantization_errors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timing': {
                'total_time': self.total_time,
                'layer_times': self.layer_times,
                'calibration_time': self.calibration_time,
                'quantization_time': self.quantization_time,
            },
            'memory': {
                'peak_memory_gb': self.peak_memory_gb,
                'memory_saved_gb': self.memory_saved_gb,
                'memory_efficiency': self.memory_efficiency,
            },
            'model': {
                'original_size_mb': self.original_size_mb,
                'quantized_size_mb': self.quantized_size_mb,
                'compression_ratio': self.compression_ratio,
            },
            'quality': {
                'perplexity_original': self.perplexity_original,
                'perplexity_quantized': self.perplexity_quantized,
                'perplexity_degradation': self.perplexity_degradation,
            },
            'throughput': {
                'tokens_per_second': self.tokens_per_second,
                'samples_per_second': self.samples_per_second,
            },
            'errors': self.quantization_errors,
        }
    
    def save(self, path: Union[str, Path]):
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'QuantizationMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        metrics = cls()
        metrics.total_time = data['timing']['total_time']
        metrics.layer_times = data['timing']['layer_times']
        metrics.calibration_time = data['timing']['calibration_time']
        metrics.quantization_time = data['timing']['quantization_time']
        
        metrics.peak_memory_gb = data['memory']['peak_memory_gb']
        metrics.memory_saved_gb = data['memory']['memory_saved_gb']
        metrics.memory_efficiency = data['memory']['memory_efficiency']
        
        metrics.original_size_mb = data['model']['original_size_mb']
        metrics.quantized_size_mb = data['model']['quantized_size_mb']
        metrics.compression_ratio = data['model']['compression_ratio']
        
        metrics.perplexity_original = data['quality']['perplexity_original']
        metrics.perplexity_quantized = data['quality']['perplexity_quantized']
        metrics.perplexity_degradation = data['quality']['perplexity_degradation']
        
        metrics.tokens_per_second = data['throughput']['tokens_per_second']
        metrics.samples_per_second = data['throughput']['samples_per_second']
        
        metrics.quantization_errors = data['errors']
        
        return metrics


class PerformanceMonitor:
    """
    Monitor performance during quantization.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = QuantizationMetrics()
        self.start_times = {}
        self.layer_start_times = {}
        self.memory_tracker = []
        
    def start_timer(self, name: str):
        """Start a named timer."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time."""
        if name not in self.start_times:
            return 0.0
        
        elapsed = time.time() - self.start_times[name]
        del self.start_times[name]
        return elapsed
    
    def start_layer_timer(self, layer_name: str):
        """Start timing a layer quantization."""
        self.layer_start_times[layer_name] = time.time()
    
    def end_layer_timer(self, layer_name: str):
        """End timing a layer quantization."""
        if layer_name in self.layer_start_times:
            elapsed = time.time() - self.layer_start_times[layer_name]
            self.metrics.layer_times[layer_name] = elapsed
            del self.layer_start_times[layer_name]
    
    def record_memory_usage(self, device: Optional[torch.device] = None):
        """Record current memory usage."""
        if torch.cuda.is_available():
            if device is None:
                device = torch.cuda.current_device()
            
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            
            self.memory_tracker.append({
                'timestamp': time.time(),
                'allocated_gb': allocated,
                'reserved_gb': reserved
            })
            
            # Update peak memory
            if allocated > self.metrics.peak_memory_gb:
                self.metrics.peak_memory_gb = allocated
    
    def calculate_model_size(self, model: torch.nn.Module) -> float:
        """Calculate model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        # Estimate as float32 for original models
        size_mb = total_params * 4 / 1024**2
        return size_mb
    
    def calculate_quantized_size(self, model: torch.nn.Module, bits: int = 4) -> float:
        """Calculate quantized model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        # Approximate quantized size
        size_mb = total_params * bits / 8 / 1024**2
        return size_mb
    
    def record_model_metrics(self, original_model: torch.nn.Module, quantized_model: torch.nn.Module, bits: int = 4):
        """Record model size metrics."""
        self.metrics.original_size_mb = self.calculate_model_size(original_model)
        self.metrics.quantized_size_mb = self.calculate_quantized_size(quantized_model, bits)
        
        if self.metrics.quantized_size_mb > 0:
            self.metrics.compression_ratio = self.metrics.original_size_mb / self.metrics.quantized_size_mb
        
        self.metrics.memory_saved_gb = (self.metrics.original_size_mb - self.metrics.quantized_size_mb) / 1024
        
        if self.metrics.original_size_mb > 0:
            self.metrics.memory_efficiency = self.metrics.memory_saved_gb / (self.metrics.original_size_mb / 1024)
    
    def calculate_perplexity(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> float:
        """
        Calculate model perplexity on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on
            
        Returns:
            Perplexity score
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    
                    # Simple forward pass (model-dependent)
                    try:
                        outputs = model(**inputs)
                        if hasattr(outputs, 'loss'):
                            loss = outputs.loss
                        elif hasattr(outputs, 'logits'):
                            # Calculate cross-entropy loss
                            logits = outputs.logits
                            labels = inputs.get('input_ids', inputs.get('labels'))
                            if labels is not None:
                                loss = torch.nn.functional.cross_entropy(
                                    logits.view(-1, logits.size(-1)),
                                    labels.view(-1),
                                    ignore_index=-100
                                )
                            else:
                                continue
                        else:
                            continue
                        
                        total_loss += loss.item()
                        total_tokens += inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])).sum().item()
                        
                    except Exception as e:
                        print(f"Error calculating perplexity: {e}")
                        continue
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity
    
    def record_perplexity(
        self,
        original_model: torch.nn.Module,
        quantized_model: torch.nn.Module,
        eval_dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ):
        """Record perplexity metrics for both models."""
        try:
            self.metrics.perplexity_original = self.calculate_perplexity(
                original_model, eval_dataloader, device
            )
            self.metrics.perplexity_quantized = self.calculate_perplexity(
                quantized_model, eval_dataloader, device
            )
            
            if self.metrics.perplexity_original and self.metrics.perplexity_quantized:
                self.metrics.perplexity_degradation = (
                    self.metrics.perplexity_quantized - self.metrics.perplexity_original
                ) / self.metrics.perplexity_original
        except Exception as e:
            print(f"Error recording perplexity: {e}")
    
    def measure_throughput(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        num_batches: int = 10
    ) -> Dict[str, float]:
        """
        Measure model throughput.
        
        Args:
            model: Model to benchmark
            dataloader: DataLoader with test data
            device: Device to run benchmark on
            num_batches: Number of batches to benchmark
            
        Returns:
            Dictionary with throughput metrics
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 3:  # 3 warmup batches
                    break
                
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    _ = model(**inputs)
        
        # Actual measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        total_tokens = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    _ = model(**inputs)
                    
                    # Count tokens and samples
                    if 'input_ids' in inputs:
                        total_tokens += inputs['input_ids'].numel()
                        total_samples += inputs['input_ids'].size(0)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed_time = time.time() - start_time
        
        throughput = {
            'tokens_per_second': total_tokens / elapsed_time if elapsed_time > 0 else 0,
            'samples_per_second': total_samples / elapsed_time if elapsed_time > 0 else 0,
        }
        
        return throughput
    
    def record_quantization_error(self, layer_name: str, error: float):
        """Record quantization error for a layer."""
        self.metrics.quantization_errors[layer_name] = error
    
    def finalize_metrics(self):
        """Finalize and calculate derived metrics."""
        # Calculate total time from layer times
        if self.metrics.layer_times:
            self.metrics.quantization_time = sum(self.metrics.layer_times.values())
        
        self.metrics.total_time = self.metrics.calibration_time + self.metrics.quantization_time
    
    def print_summary(self):
        """Print a summary of the metrics."""
        print("\n" + "="*50)
        print("QUANTIZATION PERFORMANCE SUMMARY")
        print("="*50)
        
        # Timing
        print(f"\nTiming:")
        print(f"  Total time: {self.metrics.total_time:.2f}s")
        print(f"  Calibration time: {self.metrics.calibration_time:.2f}s")
        print(f"  Quantization time: {self.metrics.quantization_time:.2f}s")
        
        # Memory
        print(f"\nMemory:")
        print(f"  Peak memory usage: {self.metrics.peak_memory_gb:.2f}GB")
        print(f"  Memory saved: {self.metrics.memory_saved_gb:.2f}GB")
        print(f"  Memory efficiency: {self.metrics.memory_efficiency:.1%}")
        
        # Model size
        print(f"\nModel Size:")
        print(f"  Original size: {self.metrics.original_size_mb:.1f}MB")
        print(f"  Quantized size: {self.metrics.quantized_size_mb:.1f}MB")
        print(f"  Compression ratio: {self.metrics.compression_ratio:.1f}x")
        
        # Quality
        if self.metrics.perplexity_original is not None:
            print(f"\nQuality:")
            print(f"  Original perplexity: {self.metrics.perplexity_original:.2f}")
            print(f"  Quantized perplexity: {self.metrics.perplexity_quantized:.2f}")
            if self.metrics.perplexity_degradation is not None:
                print(f"  Perplexity degradation: {self.metrics.perplexity_degradation:.1%}")
        
        # Throughput
        if self.metrics.tokens_per_second is not None:
            print(f"\nThroughput:")
            print(f"  Tokens per second: {self.metrics.tokens_per_second:.0f}")
            print(f"  Samples per second: {self.metrics.samples_per_second:.1f}")
        
        print("="*50)


class LayerProfiler:
    """
    Profile individual layer quantization performance.
    """
    
    def __init__(self):
        """Initialize layer profiler."""
        self.layer_profiles = {}
    
    def profile_layer(
        self,
        layer_name: str,
        original_layer: torch.nn.Module,
        quantized_layer: torch.nn.Module,
        calibration_data: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Profile a single layer's quantization.
        
        Args:
            layer_name: Name of the layer
            original_layer: Original layer
            quantized_layer: Quantized layer
            calibration_data: Calibration data for the layer
            
        Returns:
            Profile dictionary
        """
        profile = {}
        
        # Size comparison
        original_params = sum(p.numel() for p in original_layer.parameters())
        quantized_params = sum(p.numel() for p in quantized_layer.parameters())
        
        profile['original_params'] = original_params
        profile['quantized_params'] = quantized_params
        profile['compression_ratio'] = original_params / quantized_params if quantized_params > 0 else 0
        
        # Accuracy comparison (if possible)
        try:
            with torch.no_grad():
                original_output = original_layer(calibration_data)
                quantized_output = quantized_layer(calibration_data)
                
                # Calculate MSE
                mse = torch.mean((original_output - quantized_output) ** 2).item()
                profile['mse'] = mse
                
                # Calculate relative error
                relative_error = mse / torch.mean(original_output ** 2).item()
                profile['relative_error'] = relative_error
                
        except Exception as e:
            profile['error'] = str(e)
        
        self.layer_profiles[layer_name] = profile
        return profile
    
    def get_worst_layers(self, metric: str = 'relative_error', top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get the worst performing layers by a metric.
        
        Args:
            metric: Metric to sort by
            top_k: Number of top layers to return
            
        Returns:
            List of (layer_name, metric_value) tuples
        """
        layer_metrics = []
        for layer_name, profile in self.layer_profiles.items():
            if metric in profile:
                layer_metrics.append((layer_name, profile[metric]))
        
        layer_metrics.sort(key=lambda x: x[1], reverse=True)
        return layer_metrics[:top_k]
