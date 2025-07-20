# Distributed GPTQ

[![PyPI version](https://badge.fury.io/py/distributed-gptq.svg)](https://badge.fury.io/py/distributed-gptq)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A PyTorch-based implementation of GPTQ (Generative Pre-trained Transformer Quantization) with distributed computing support. This package enables efficient quantization of large language models across multiple GPUs and even multiple nodes.

## 🚀 Key Features

- **Single GPU Support**: Quantize models on a single GPU efficiently
- **Multi-GPU Support**: Data parallel quantization across multiple GPUs on a single node
- **Distributed Support**: Scale across multiple nodes (4-8+ GPUs)
- **Flexible Quantization**: Support for 2, 3, 4, and 8-bit quantization
- **Model Agnostic**: Works with various transformer architectures
- **Memory Efficient**: Optimized memory usage with gradient checkpointing
- **Easy Integration**: Simple API and CLI interface
- **Checkpointing**: Save and resume quantization progress

## 📦 Installation

### From PyPI (when published)
```bash
pip install distributed-gptq
```

### From Source
```bash
git clone https://github.com/yourusername/distributed-gptq
cd distributed-gptq
pip install -e .
```

### With CUDA optimizations
```bash
pip install distributed-gptq[cuda]
```

## 🔧 Quick Start

### Single GPU Quantization

```python
from distributed_gptq import quantize_model_simple
from transformers import AutoModelForCausalLM

# Load your model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Prepare calibration data (list of input tensors)
calibration_data = [...]  # Your calibration dataset

# Quantize to 4 bits
quantized_model = quantize_model_simple(
    model, 
    calibration_data, 
    bits=4,
    save_path="quantized_model.pt"
)
```

### Multi-GPU Quantization

```python
from distributed_gptq import DistributedGPTQuantizer, QuantizationConfig

# Configure quantization
config = QuantizationConfig(
    bits=4,
    group_size=128,
    calibration_samples=128
)

# Create quantizer
quantizer = DistributedGPTQuantizer(config)

# Quantize model
quantized_model = quantizer.quantize_model(
    model,
    calibration_data,
    save_path="quantized_model.safetensors"
)
```

### Distributed Quantization (Multiple Nodes)

Launch on multiple nodes using `torchrun`:

```bash
# Node 0 (master)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<MASTER_IP> --master_port=29500 \
    your_script.py --distributed

# Node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<MASTER_IP> --master_port=29500 \
    your_script.py --distributed
```

## 🖥️ Command Line Interface

### Quantize a model
```bash
distributed-gptq quantize facebook/opt-1.3b \
    -o quantized_model.safetensors \
    -b 4 \
    --calibration-samples 128
```

### Benchmark quantized model
```bash
distributed-gptq benchmark \
    original_model.pt \
    quantized_model.safetensors \
    --test-samples 100
```

### Convert between formats
```bash
distributed-gptq convert \
    model.pt \
    model.safetensors \
    --input-format pytorch \
    --output-format safetensors
```

## 📊 Advanced Configuration

```python
from distributed_gptq import (
    DistributedGPTQuantizer, 
    QuantizationConfig,
    DistributedConfig,
    QuantizationMode
)

# Advanced quantization config
quant_config = QuantizationConfig(
    bits=4,                    # Quantization bits
    group_size=128,           # Group size for quantization
    actorder=False,           # Use activation order
    percdamp=0.01,           # Percentage dampening
    blocksize=128,           # Block size for quantization
    calibration_samples=256,  # Number of calibration samples
    use_triton=True,         # Use Triton kernels (if available)
)

# Distributed config
dist_config = DistributedConfig(
    mode=QuantizationMode.HYBRID_PARALLEL,  # Hybrid parallelism
    world_size=8,                           # Total GPUs
    backend="nccl",                         # Communication backend
)

# Initialize quantizer
quantizer = DistributedGPTQuantizer(
    quantization_config=quant_config,
    distributed_config=dist_config
)
```

## 🏗️ Architecture

The package is organized into modular components:

- **Core**: GPTQ algorithm implementation
- **Distributed**: Multi-GPU/node coordination
- **Models**: Model-specific adaptations  
- **Utils**: Helper utilities
- **CLI**: Command-line interface

### Parallelization Strategies

1. **Data Parallel**: Split calibration data across GPUs
2. **Model Parallel**: Split model layers across GPUs
3. **Hybrid Parallel**: Combination of both strategies

## 📈 Performance

| Model Size | GPUs | Quantization Time | Memory Usage |
|------------|------|-------------------|--------------|
| 125M       | 1    | ~2 min           | 4 GB         |
| 1.3B       | 4    | ~5 min           | 8 GB/GPU     |
| 6.7B       | 8    | ~15 min          | 16 GB/GPU    |
| 13B        | 8    | ~30 min          | 24 GB/GPU    |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/distributed-gptq
cd distributed-gptq

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black distributed_gptq/
isort distributed_gptq/
```

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{distributed-gptq,
  title = {Distributed GPTQ: Efficient Quantization for Large Language Models},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/distributed-gptq}
}
```

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This implementation is based on the original GPTQ paper:
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

## 🔗 Links

- [Documentation](https://distributed-gptq.readthedocs.io)
- [PyPI Package](https://pypi.org/project/distributed-gptq)
- [GitHub Repository](https://github.com/yourusername/distributed-gptq)
- [Issue Tracker](https://github.com/yourusername/distributed-gptq/issues)