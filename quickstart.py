"""
Quick start script for distributed GPTQ.

This script demonstrates basic usage of the distributed GPTQ package
with a simple example model.
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

def check_environment():
    """Check if the environment is set up correctly."""
    print("🔍 Checking environment...")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} detected")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        else:
            print("⚠️  CUDA not available - will use CPU")
            
    except ImportError:
        print("❌ PyTorch not found. Please install with: pip install torch")
        return False
    
    # Check transformers
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} detected")
    except ImportError:
        print("❌ Transformers not found. Please install with: pip install transformers")
        return False
    
    # Check distributed GPTQ
    try:
        import distributed_gptq
        print(f"✅ Distributed GPTQ {distributed_gptq.__version__} detected")
    except ImportError:
        print("❌ Distributed GPTQ not found. Please install with: pip install -e .")
        return False
    
    return True


def create_simple_model():
    """Create a simple test model."""
    print("\n🏗️  Creating test model...")
    
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(), 
        nn.Linear(512, 256),
    )
    
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def create_calibration_data(model, num_samples=50):
    """Create calibration data for the model."""
    print(f"\n📊 Creating {num_samples} calibration samples...")
    
    calibration_data = []
    model.eval()
    
    for i in range(num_samples):
        # Create random input
        input_tensor = torch.randn(1, 512)
        calibration_data.append(input_tensor)
    
    print(f"✅ Created {len(calibration_data)} calibration samples")
    return calibration_data


def run_simple_quantization():
    """Run a simple quantization example."""
    print("\n🚀 Running simple quantization example...")
    
    try:
        from distributed_gptq import quantize_model_simple, QuantizationConfig
        
        # Create model and data
        model = create_simple_model()
        calibration_data = create_calibration_data(model)
        
        # Show original model size
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        print(f"📏 Original model size: {original_size:.2f} MB")
        
        # Quantize the model
        print("\n⚙️  Starting quantization...")
        start_time = time.time()
        
        quantized_model = quantize_model_simple(
            model=model,
            calibration_data=calibration_data,
            bits=4,
            save_path=None  # Don't save for this demo
        )
        
        quantization_time = time.time() - start_time
        print(f"✅ Quantization completed in {quantization_time:.2f} seconds")
        
        # Test inference
        print("\n🧪 Testing inference...")
        test_input = torch.randn(1, 512)
        
        # Original model inference
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            original_output = model(test_input)
            original_time = time.time() - start_time
        
        # Quantized model inference  
        quantized_model.eval()
        with torch.no_grad():
            start_time = time.time()
            quantized_output = quantized_model(test_input)
            quantized_time = time.time() - start_time
        
        # Calculate metrics
        mse = torch.mean((original_output - quantized_output) ** 2).item()
        speedup = original_time / quantized_time if quantized_time > 0 else 1.0
        
        print(f"✅ Original inference time: {original_time*1000:.2f}ms")
        print(f"✅ Quantized inference time: {quantized_time*1000:.2f}ms")
        print(f"✅ Speedup: {speedup:.2f}x")
        print(f"✅ MSE between outputs: {mse:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_benchmark_example():
    """Run a benchmark example."""
    print("\n📈 Running benchmark example...")
    
    try:
        from distributed_gptq.utils.benchmark import quick_benchmark
        
        # Create models
        original_model = create_simple_model()
        quantized_model = create_simple_model()  # Simulate quantized model
        
        # Run benchmark
        test_input = torch.randn(1, 512)
        
        print("⚙️  Running benchmark...")
        results = quick_benchmark(
            original_model=original_model,
            quantized_model=quantized_model,
            test_input=test_input,
            iterations=10
        )
        
        print("📊 Benchmark Results:")
        for metric, value in results.items():
            print(f"   {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def show_cli_help():
    """Show CLI usage examples."""
    print("\n🖥️  CLI Usage Examples:")
    print("=" * 50)
    
    examples = [
        "# Quantize a model",
        "distributed-gptq quantize facebook/opt-125m -o quantized_model.safetensors -b 4",
        "",
        "# Benchmark models", 
        "distributed-gptq benchmark original.pt quantized.safetensors",
        "",
        "# Convert model formats",
        "distributed-gptq convert model.pt model.safetensors",
        "",
        "# Multi-GPU quantization",
        "torchrun --nproc_per_node=4 -m distributed_gptq.cli quantize \\",
        "    facebook/opt-1.3b -o quantized_opt_1.3b -b 4 --distributed",
    ]
    
    for example in examples:
        print(example)


def main():
    """Main function."""
    print("🎯 Distributed GPTQ Quick Start")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please install required dependencies.")
        return 1
    
    # Run examples
    examples = [
        ("Simple Quantization", run_simple_quantization),
        ("Benchmark Example", run_benchmark_example),
    ]
    
    for name, func in examples:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        
        try:
            success = func()
            if success:
                print(f"✅ {name} completed successfully!")
            else:
                print(f"❌ {name} failed!")
        except Exception as e:
            print(f"💥 {name} crashed: {e}")
    
    # Show CLI help
    show_cli_help()
    
    print(f"\n{'='*60}")
    print("🎉 Quick start completed!")
    print("📚 Check the examples/ directory for more advanced usage")
    print("📖 Read README.md for full documentation")
    print('='*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
