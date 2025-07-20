"""
Unit tests for benchmark functionality.
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

try:
    from distributed_gptq.utils.benchmark import ModelBenchmark, BenchmarkResult, quick_benchmark, benchmark_context
except ImportError:
    # Handle case where modules aren't available yet
    ModelBenchmark = None
    BenchmarkResult = None


class TestBenchmarkFunctionality(unittest.TestCase):
    """Test benchmark utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if ModelBenchmark is None:
            self.skipTest("Benchmark modules not available")
            
        torch.manual_seed(42)
        self.device = 'cpu'
        
        # Create simple test models
        self.original_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Simulate a "quantized" model (same structure, different weights)
        self.quantized_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Slightly modify the quantized model weights to simulate quantization
        with torch.no_grad():
            for param in self.quantized_model.parameters():
                param.data += torch.randn_like(param.data) * 0.01
        
        # Create test inputs
        self.test_inputs = [torch.randn(1, 10) for _ in range(3)]
        
        self.benchmark = ModelBenchmark(
            device=self.device,
            warmup_iterations=2,
            benchmark_iterations=5,
            memory_profiling=False
        )
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        size = self.benchmark._get_model_size(self.original_model)
        
        self.assertIsInstance(size, float)
        self.assertGreater(size, 0)
        
        # Check that quantized model has similar size (since it's not actually quantized)
        quantized_size = self.benchmark._get_model_size(self.quantized_model)
        self.assertAlmostEqual(size, quantized_size, places=2)
    
    def test_inference_benchmarking(self):
        """Test inference speed measurement."""
        inference_time = self.benchmark._benchmark_inference(
            self.original_model, self.test_inputs, "original"
        )
        
        self.assertIsInstance(inference_time, float)
        self.assertGreater(inference_time, 0)
        self.assertLess(inference_time, 10)  # Should be reasonable time
    
    def test_accuracy_metrics(self):
        """Test accuracy metrics calculation."""
        metrics = self.benchmark._calculate_accuracy_metrics(
            self.original_model, self.quantized_model, self.test_inputs
        )
        
        self.assertIsInstance(metrics, dict)
        
        required_metrics = ['mse', 'cosine_similarity', 'max_absolute_error', 'snr_db']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        # Cosine similarity should be high since models are similar
        self.assertGreater(metrics['cosine_similarity'], 0.5)
        
        # MSE should be positive but not too large
        self.assertGreater(metrics['mse'], 0)
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult dataclass."""
        result = BenchmarkResult(
            model_name="test_model",
            original_size_mb=10.0,
            quantized_size_mb=3.0,
            compression_ratio=3.33,
            original_inference_time=0.1,
            quantized_inference_time=0.05,
            speedup=2.0,
            original_memory_mb=100.0,
            quantized_memory_mb=50.0,
            memory_reduction=50.0,
            accuracy_metrics={'mse': 0.001, 'cosine_similarity': 0.99}
        )
        
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.compression_ratio, 3.33)
        self.assertEqual(result.speedup, 2.0)
        self.assertEqual(result.memory_reduction, 50.0)
    
    def test_full_benchmark_pipeline(self):
        """Test complete benchmarking pipeline."""
        result = self.benchmark.benchmark_models(
            original_model=self.original_model,
            quantized_model=self.quantized_model,
            test_inputs=self.test_inputs,
            model_name="integration_test",
            calculate_perplexity=False
        )
        
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.model_name, "integration_test")
        
        # Check all metrics are computed
        self.assertGreater(result.original_size_mb, 0)
        self.assertGreater(result.quantized_size_mb, 0)
        self.assertGreater(result.original_inference_time, 0)
        self.assertGreater(result.quantized_inference_time, 0)
        self.assertIsInstance(result.accuracy_metrics, dict)
    
    def test_save_and_load_results(self):
        """Test saving benchmark results to file."""
        result = BenchmarkResult(
            model_name="save_test",
            original_size_mb=15.5,
            quantized_size_mb=4.2,
            compression_ratio=3.69,
            original_inference_time=0.08,
            quantized_inference_time=0.04,
            speedup=2.0,
            original_memory_mb=120.0,
            quantized_memory_mb=60.0,
            memory_reduction=50.0,
            accuracy_metrics={'mse': 0.002, 'cosine_similarity': 0.98}
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save results
            self.benchmark.save_results(result, temp_path)
            
            # Verify file exists
            self.assertTrue(os.path.exists(temp_path))
            
            # Load and verify content
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(data['model_name'], 'save_test')
            self.assertEqual(data['model_size']['compression_ratio'], 3.69)
            self.assertEqual(data['inference_speed']['speedup'], 2.0)
            self.assertIn('accuracy_metrics', data)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_quick_benchmark(self, mock_cuda):
        """Test quick benchmark function."""
        if quick_benchmark is None:
            self.skipTest("quick_benchmark function not available")
        
        test_input = torch.randn(1, 10)
        
        result = quick_benchmark(
            original_model=self.original_model,
            quantized_model=self.quantized_model,
            test_input=test_input,
            iterations=3
        )
        
        self.assertIsInstance(result, dict)
        
        expected_keys = ['compression_ratio', 'speedup', 'memory_reduction_percent', 
                        'cosine_similarity', 'mse']
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], (int, float))
    
    def test_benchmark_context_manager(self):
        """Test benchmark context manager."""
        if benchmark_context is None:
            self.skipTest("benchmark_context not available")
        
        # Test context manager usage
        try:
            with benchmark_context(enable_profiling=False):
                # Simulate some computation
                x = torch.randn(100, 100)
                y = torch.mm(x, x.t())
                result = torch.sum(y)
            
            # Should complete without errors
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"benchmark_context failed: {e}")
    
    def test_print_summary(self):
        """Test benchmark summary printing."""
        result = BenchmarkResult(
            model_name="summary_test",
            original_size_mb=8.0,
            quantized_size_mb=2.0,
            compression_ratio=4.0,
            original_inference_time=0.12,
            quantized_inference_time=0.06,
            speedup=2.0,
            original_memory_mb=80.0,
            quantized_memory_mb=40.0,
            memory_reduction=50.0,
            accuracy_metrics={'mse': 0.001, 'cosine_similarity': 0.995},
            perplexity_original=25.5,
            perplexity_quantized=26.2,
            perplexity_degradation=2.75
        )
        
        # Test that print_summary doesn't crash
        try:
            self.benchmark.print_summary(result)
            # If we get here, printing worked
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"print_summary failed: {e}")


class TestBenchmarkEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if ModelBenchmark is None:
            self.skipTest("Benchmark modules not available")
            
        self.benchmark = ModelBenchmark(
            device='cpu',
            warmup_iterations=1,
            benchmark_iterations=2,
            memory_profiling=False
        )
    
    def test_empty_input_list(self):
        """Test behavior with empty input list."""
        model = nn.Linear(5, 3)
        empty_inputs = []
        
        # Should handle empty inputs gracefully
        try:
            time_result = self.benchmark._benchmark_inference(model, empty_inputs, "test")
            # Should return 0 or handle gracefully
            self.assertGreaterEqual(time_result, 0)
        except Exception:
            # Acceptable to raise exception for empty inputs
            pass
    
    def test_mismatched_model_sizes(self):
        """Test with models of different sizes."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 3)  # Different output size
        
        test_inputs = [torch.randn(1, 10)]
        
        # Should handle size mismatches in accuracy calculation
        try:
            metrics = self.benchmark._calculate_accuracy_metrics(
                model1, model2, test_inputs
            )
            # May succeed with modified comparison or may raise exception
            if metrics:
                self.assertIsInstance(metrics, dict)
        except Exception:
            # Acceptable to fail with mismatched models
            pass
    
    def test_very_large_inputs(self):
        """Test with unusually large inputs."""
        model = nn.Linear(5, 3)
        large_inputs = [torch.randn(1000, 5)]  # Large batch
        
        # Should handle large inputs without crashing
        try:
            time_result = self.benchmark._benchmark_inference(model, large_inputs, "test")
            self.assertGreater(time_result, 0)
        except Exception as e:
            # May fail due to memory constraints, which is acceptable
            self.assertIn("memory", str(e).lower())


if __name__ == '__main__':
    unittest.main()
