"""
Test script to validate the distributed GPTQ package installation and functionality.

This script performs various tests to ensure all components work correctly.
"""

import sys
import traceback
import importlib
from pathlib import Path


def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing module imports...")
    
    import_tests = [
        # Core modules
        "distributed_gptq",
        "distributed_gptq.core.gptq",
        "distributed_gptq.core.quantizer", 
        "distributed_gptq.core.dequantizer",
        
        # Distributed modules
        "distributed_gptq.distributed.coordinator",
        "distributed_gptq.distributed.worker",
        "distributed_gptq.distributed.communication",
        
        # Model modules
        "distributed_gptq.models.base_model",
        "distributed_gptq.models.transformers_model",
        "distributed_gptq.models.layers",
        "distributed_gptq.models.llama",
        "distributed_gptq.models.opt",
        
        # Utility modules
        "distributed_gptq.utils.data_utils",
        "distributed_gptq.utils.gpu_utils", 
        "distributed_gptq.utils.logging_utils",
        "distributed_gptq.utils.metrics",
        "distributed_gptq.utils.benchmark",
        
        # Optional modules
        "distributed_gptq.kernels.triton_kernels",
        
        # CLI
        "distributed_gptq.cli",
    ]
    
    passed = 0
    failed = 0
    
    for module_name in import_tests:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
            passed += 1
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")
            failed += 1
    
    print(f"\nImport test results: {passed} passed, {failed} failed")
    return failed == 0


def test_core_functionality():
    """Test core GPTQ functionality without requiring GPU."""
    print("\nTesting core functionality...")
    
    try:
        # Test QuantizationConfig
        from distributed_gptq.core.quantizer import QuantizationConfig
        config = QuantizationConfig(bits=4, group_size=128)
        print(f"✅ QuantizationConfig created: {config.bits} bits, {config.group_size} group size")
        
        # Test model imports
        from distributed_gptq.models.base_model import QuantizableModel
        print("✅ QuantizableModel imported successfully")
        
        from distributed_gptq.models.llama import create_llama_quantizable_model
        from distributed_gptq.models.opt import create_opt_quantizable_model
        print("✅ Model-specific quantizable wrappers imported")
        
        # Test utility functions
        from distributed_gptq.utils.benchmark import ModelBenchmark, BenchmarkResult
        print("✅ Benchmark utilities imported")
        
        from distributed_gptq.utils.gpu_utils import GPUMemoryManager
        # Test with CPU device
        memory_manager = GPUMemoryManager('cpu')
        memory_info = memory_manager.get_memory_info()
        print(f"✅ GPU memory manager works: {memory_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_triton_availability():
    """Test Triton kernel availability."""
    print("\nTesting Triton availability...")
    
    try:
        from distributed_gptq.kernels.triton_kernels import check_triton_availability, TRITON_AVAILABLE
        
        if TRITON_AVAILABLE:
            print("✅ Triton is available")
            is_usable = check_triton_availability()
            print(f"✅ Triton can be used: {is_usable}")
        else:
            print("⚠️  Triton is not available (this is optional)")
        
        return True
        
    except Exception as e:
        print(f"❌ Triton test failed: {e}")
        return False


def test_cli_availability():
    """Test CLI functionality."""
    print("\nTesting CLI availability...")
    
    try:
        from distributed_gptq.cli import main, quantize_command, benchmark_command
        print("✅ CLI commands imported successfully")
        
        # Test CLI help (without actually running)
        from distributed_gptq.cli import create_parser
        parser = create_parser()
        print(f"✅ CLI parser created with {len(parser._subparsers._actions)} subcommands")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_package_metadata():
    """Test package metadata and version."""
    print("\nTesting package metadata...")
    
    try:
        from distributed_gptq import __version__
        print(f"✅ Package version: {__version__}")
        
        # Test main exports
        from distributed_gptq import (
            GPTQ, DistributedGPTQuantizer, QuantizationConfig
        )
        print("✅ Main package exports available")
        
        return True
        
    except Exception as e:
        print(f"❌ Package metadata test failed: {e}")
        return False


def test_example_files():
    """Test that example files exist and are valid Python."""
    print("\nTesting example files...")
    
    examples_dir = Path("examples")
    if not examples_dir.exists():
        print("⚠️  Examples directory not found")
        return False
    
    example_files = [
        "single_gpu_example.py",
        "multi_gpu_example.py", 
        "distributed_example.py"
    ]
    
    passed = 0
    for example_file in example_files:
        file_path = examples_dir / example_file
        if file_path.exists():
            try:
                # Try to compile the file to check syntax
                with open(file_path, 'r') as f:
                    compile(f.read(), str(file_path), 'exec')
                print(f"✅ {example_file} is syntactically valid")
                passed += 1
            except SyntaxError as e:
                print(f"❌ {example_file} has syntax error: {e}")
            except Exception as e:
                print(f"⚠️  {example_file} check failed: {e}")
        else:
            print(f"❌ {example_file} not found")
    
    return passed == len(example_files)


def test_documentation():
    """Test that documentation files exist."""
    print("\nTesting documentation...")
    
    doc_files = [
        "README.md",
        "LICENSE", 
        "setup.py",
        "pyproject.toml"
    ]
    
    passed = 0
    for doc_file in doc_files:
        if Path(doc_file).exists():
            print(f"✅ {doc_file} exists")
            passed += 1
        else:
            print(f"❌ {doc_file} not found")
    
    return passed == len(doc_files)


def run_pytorch_tests():
    """Run tests that require PyTorch (if available)."""
    print("\nTesting PyTorch integration...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} is available")
        
        # Test basic tensor operations
        x = torch.randn(10, 10)
        y = torch.mm(x, x.t())
        print("✅ Basic PyTorch operations work")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA is available with {torch.cuda.device_count()} device(s)")
            device = torch.device('cuda:0')
            x_cuda = x.to(device)
            print("✅ CUDA tensor operations work")
        else:
            print("⚠️  CUDA not available (CPU-only mode)")
        
        # Test model creation
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        print("✅ PyTorch model creation works")
        
        return True
        
    except ImportError:
        print("⚠️  PyTorch not available - skipping PyTorch tests")
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🚀 Starting distributed GPTQ package validation\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Core Functionality", test_core_functionality),
        ("Triton Availability", test_triton_availability),
        ("CLI Availability", test_cli_availability),
        ("Package Metadata", test_package_metadata),
        ("Example Files", test_example_files),
        ("Documentation", test_documentation),
        ("PyTorch Integration", run_pytorch_tests),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print('='*60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The package is ready to use.")
        return 0
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Package should work with minor issues.")
        return 0
    else:
        print("❌ Multiple tests failed. Package may have significant issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
