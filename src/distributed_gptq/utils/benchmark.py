"""
Performance benchmarking utilities for distributed GPTQ.

This module provides utilities for benchmarking quantized models against
their original counterparts, measuring inference speed, memory usage,
and accuracy metrics.
"""

import torch
import torch.nn as nn
import time
import gc
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import psutil
from contextlib import contextmanager

from .gpu_utils import GPUMemoryManager
from .metrics import QuantizationMetrics


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results.
    """
    model_name: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    original_inference_time: float
    quantized_inference_time: float
    speedup: float
    original_memory_mb: float
    quantized_memory_mb: float
    memory_reduction: float
    accuracy_metrics: Dict[str, float]
    perplexity_original: Optional[float] = None
    perplexity_quantized: Optional[float] = None
    perplexity_degradation: Optional[float] = None


class ModelBenchmark:
    """
    Comprehensive benchmarking suite for quantized models.
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = None,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        memory_profiling: bool = True
    ):
        """
        Initialize the benchmark suite.
        
        Args:
            device: Device to run benchmarks on
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            memory_profiling: Whether to profile memory usage
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.memory_profiling = memory_profiling
        
        if memory_profiling and torch.cuda.is_available():
            self.memory_manager = GPUMemoryManager(self.device)
        else:
            self.memory_manager = None
    
    def benchmark_models(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_inputs: List[torch.Tensor],
        model_name: str = "model",
        calculate_perplexity: bool = True,
        target_outputs: Optional[List[torch.Tensor]] = None
    ) -> BenchmarkResult:
        """
        Comprehensive benchmark comparing original and quantized models.
        
        Args:
            original_model: Original unquantized model
            quantized_model: Quantized model
            test_inputs: List of test input tensors
            model_name: Name for the benchmark results
            calculate_perplexity: Whether to calculate perplexity
            target_outputs: Ground truth outputs for accuracy calculation
            
        Returns:
            BenchmarkResult containing all metrics
        """
        logger.info(f"Starting comprehensive benchmark for {model_name}")
        
        # Move models to device
        original_model = original_model.to(self.device)
        quantized_model = quantized_model.to(self.device)
        
        # Model size comparison
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        
        # Inference speed benchmark
        original_time = self._benchmark_inference(original_model, test_inputs, "original")
        quantized_time = self._benchmark_inference(quantized_model, test_inputs, "quantized")
        speedup = original_time / quantized_time if quantized_time > 0 else 0
        
        # Memory usage benchmark
        original_memory, quantized_memory = self._benchmark_memory(
            original_model, quantized_model, test_inputs
        )
        memory_reduction = (original_memory - quantized_memory) / original_memory * 100
        
        # Accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(
            original_model, quantized_model, test_inputs, target_outputs
        )
        
        # Perplexity calculation (for language models)
        perplexity_original = None
        perplexity_quantized = None
        perplexity_degradation = None
        
        if calculate_perplexity:
            try:
                perplexity_original = self._calculate_perplexity(original_model, test_inputs)
                perplexity_quantized = self._calculate_perplexity(quantized_model, test_inputs)
                if perplexity_original and perplexity_quantized:
                    perplexity_degradation = (perplexity_quantized - perplexity_original) / perplexity_original * 100
            except Exception as e:
                logger.warning(f"Could not calculate perplexity: {e}")
        
        return BenchmarkResult(
            model_name=model_name,
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=compression_ratio,
            original_inference_time=original_time,
            quantized_inference_time=quantized_time,
            speedup=speedup,
            original_memory_mb=original_memory,
            quantized_memory_mb=quantized_memory,
            memory_reduction=memory_reduction,
            accuracy_metrics=accuracy_metrics,
            perplexity_original=perplexity_original,
            perplexity_quantized=perplexity_quantized,
            perplexity_degradation=perplexity_degradation
        )
    
    def _get_model_size(self, model: nn.Module) -> float:
        """
        Calculate model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _benchmark_inference(
        self,
        model: nn.Module,
        test_inputs: List[torch.Tensor],
        model_type: str
    ) -> float:
        """
        Benchmark inference speed.
        
        Args:
            model: Model to benchmark
            test_inputs: Test input tensors
            model_type: Type description for logging
            
        Returns:
            Average inference time in seconds
        """
        model.eval()
        times = []
        
        # Move inputs to device
        test_inputs = [inp.to(self.device) for inp in test_inputs]
        
        # Warmup
        logger.info(f"Warming up {model_type} model...")
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                for inp in test_inputs:
                    _ = model(inp)
        
        # Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        logger.info(f"Benchmarking {model_type} model inference...")
        with torch.no_grad():
            for i in range(self.benchmark_iterations):
                start_time = time.perf_counter()
                
                for inp in test_inputs:
                    _ = model(inp)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        logger.info(f"{model_type} inference time: {avg_time:.4f}±{std_time:.4f}s")
        return avg_time
    
    def _benchmark_memory(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_inputs: List[torch.Tensor]
    ) -> Tuple[float, float]:
        """
        Benchmark memory usage during inference.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_inputs: Test inputs
            
        Returns:
            Tuple of (original_memory_mb, quantized_memory_mb)
        """
        if not self.memory_profiling or not torch.cuda.is_available():
            return 0.0, 0.0
        
        test_inputs = [inp.to(self.device) for inp in test_inputs]
        
        # Benchmark original model memory
        torch.cuda.empty_cache()
        gc.collect()
        
        original_model.eval()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            for inp in test_inputs:
                _ = original_model(inp)
        
        original_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Clear memory
        del original_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark quantized model memory
        quantized_model.eval()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            for inp in test_inputs:
                _ = quantized_model(inp)
        
        quantized_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        logger.info(f"Memory usage - Original: {original_memory:.2f}MB, Quantized: {quantized_memory:.2f}MB")
        return original_memory, quantized_memory
    
    def _calculate_accuracy_metrics(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_inputs: List[torch.Tensor],
        target_outputs: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Calculate accuracy metrics comparing models.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_inputs: Test inputs
            target_outputs: Ground truth outputs
            
        Returns:
            Dictionary of accuracy metrics
        """
        metrics = {}
        test_inputs = [inp.to(self.device) for inp in test_inputs]
        
        original_model.eval()
        quantized_model.eval()
        
        mse_errors = []
        cosine_similarities = []
        max_abs_errors = []
        
        with torch.no_grad():
            for i, inp in enumerate(test_inputs):
                orig_out = original_model(inp)
                quant_out = quantized_model(inp)
                
                # Handle different output types
                if isinstance(orig_out, tuple):
                    orig_out = orig_out[0]  # Take logits
                if isinstance(quant_out, tuple):
                    quant_out = quant_out[0]
                
                # Flatten for comparison
                orig_flat = orig_out.flatten()
                quant_flat = quant_out.flatten()
                
                # MSE
                mse = torch.mean((orig_flat - quant_flat) ** 2).item()
                mse_errors.append(mse)
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_flat.unsqueeze(0), quant_flat.unsqueeze(0)
                ).item()
                cosine_similarities.append(cos_sim)
                
                # Max absolute error
                max_abs_error = torch.max(torch.abs(orig_flat - quant_flat)).item()
                max_abs_errors.append(max_abs_error)
        
        metrics['mse'] = np.mean(mse_errors)
        metrics['cosine_similarity'] = np.mean(cosine_similarities)
        metrics['max_absolute_error'] = np.mean(max_abs_errors)
        metrics['snr_db'] = 10 * np.log10(np.var([out.cpu().numpy() for out in test_inputs]) / np.mean(mse_errors)) if np.mean(mse_errors) > 0 else float('inf')
        
        return metrics
    
    def _calculate_perplexity(
        self,
        model: nn.Module,
        test_inputs: List[torch.Tensor]
    ) -> Optional[float]:
        """
        Calculate perplexity for language models.
        
        Args:
            model: Language model
            test_inputs: Test input sequences
            
        Returns:
            Perplexity score or None if calculation fails
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for inp in test_inputs:
                inp = inp.to(self.device)
                
                # For language models, we use input as both input and target (shifted)
                if inp.dim() == 2:  # [batch_size, seq_len]
                    inputs = inp[:, :-1]
                    targets = inp[:, 1:]
                else:
                    continue  # Skip if not a sequence
                
                try:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Reshape for loss calculation
                    logits = logits.reshape(-1, logits.size(-1))
                    targets = targets.reshape(-1)
                    
                    loss = criterion(logits, targets)
                    total_loss += loss.item()
                    total_tokens += targets.numel()
                    
                except Exception as e:
                    logger.warning(f"Error calculating perplexity for batch: {e}")
                    continue
        
        if total_tokens == 0:
            return None
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def save_results(self, result: BenchmarkResult, output_path: Union[str, Path]):
        """
        Save benchmark results to file.
        
        Args:
            result: Benchmark result to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        # Convert to dictionary
        result_dict = {
            'model_name': result.model_name,
            'model_size': {
                'original_mb': result.original_size_mb,
                'quantized_mb': result.quantized_size_mb,
                'compression_ratio': result.compression_ratio
            },
            'inference_speed': {
                'original_time_s': result.original_inference_time,
                'quantized_time_s': result.quantized_inference_time,
                'speedup': result.speedup
            },
            'memory_usage': {
                'original_mb': result.original_memory_mb,
                'quantized_mb': result.quantized_memory_mb,
                'reduction_percent': result.memory_reduction
            },
            'accuracy_metrics': result.accuracy_metrics,
            'perplexity': {
                'original': result.perplexity_original,
                'quantized': result.perplexity_quantized,
                'degradation_percent': result.perplexity_degradation
            }
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")
    
    def print_summary(self, result: BenchmarkResult):
        """
        Print a formatted summary of benchmark results.
        
        Args:
            result: Benchmark result to summarize
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS: {result.model_name}")
        print(f"{'='*60}")
        
        print(f"\n📊 MODEL SIZE:")
        print(f"  Original:    {result.original_size_mb:.2f} MB")
        print(f"  Quantized:   {result.quantized_size_mb:.2f} MB")
        print(f"  Compression: {result.compression_ratio:.2f}x")
        
        print(f"\n⚡ INFERENCE SPEED:")
        print(f"  Original:    {result.original_inference_time:.4f}s")
        print(f"  Quantized:   {result.quantized_inference_time:.4f}s")
        print(f"  Speedup:     {result.speedup:.2f}x")
        
        print(f"\n💾 MEMORY USAGE:")
        print(f"  Original:    {result.original_memory_mb:.2f} MB")
        print(f"  Quantized:   {result.quantized_memory_mb:.2f} MB")
        print(f"  Reduction:   {result.memory_reduction:.1f}%")
        
        print(f"\n🎯 ACCURACY METRICS:")
        for metric, value in result.accuracy_metrics.items():
            print(f"  {metric:20s}: {value:.6f}")
        
        if result.perplexity_original is not None:
            print(f"\n📈 PERPLEXITY:")
            print(f"  Original:    {result.perplexity_original:.4f}")
            print(f"  Quantized:   {result.perplexity_quantized:.4f}")
            print(f"  Degradation: {result.perplexity_degradation:.2f}%")
        
        print(f"\n{'='*60}")


@contextmanager
def benchmark_context(enable_profiling: bool = True):
    """
    Context manager for benchmarking with proper cleanup.
    
    Args:
        enable_profiling: Whether to enable memory profiling
    """
    if enable_profiling and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def quick_benchmark(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_input: torch.Tensor,
    iterations: int = 100
) -> Dict[str, float]:
    """
    Quick benchmark for basic performance comparison.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        test_input: Single test input
        iterations: Number of benchmark iterations
        
    Returns:
        Dictionary with basic performance metrics
    """
    device = next(original_model.parameters()).device
    test_input = test_input.to(device)
    
    benchmark = ModelBenchmark(
        device=device,
        warmup_iterations=10,
        benchmark_iterations=iterations,
        memory_profiling=torch.cuda.is_available()
    )
    
    result = benchmark.benchmark_models(
        original_model=original_model,
        quantized_model=quantized_model,
        test_inputs=[test_input],
        model_name="quick_benchmark",
        calculate_perplexity=False
    )
    
    return {
        'compression_ratio': result.compression_ratio,
        'speedup': result.speedup,
        'memory_reduction_percent': result.memory_reduction,
        'cosine_similarity': result.accuracy_metrics.get('cosine_similarity', 0.0),
        'mse': result.accuracy_metrics.get('mse', 0.0)
    }
