# Distributed GPTQ Examples

This directory contains examples demonstrating different usage patterns of the distributed-gptq package.

## Examples Overview

### 1. `single_gpu_example.py` - Single GPU Quantization
Demonstrates basic quantization on a single GPU using a small model.

**Usage:**
```bash
python single_gpu_example.py
```

**Features:**
- Uses facebook/opt-125m (small model for quick testing)
- Creates sample calibration data
- Performs 4-bit quantization
- Tests the quantized model with text generation

### 2. `multi_gpu_example.py` - Multi-GPU Quantization
Shows how to use multiple GPUs for faster quantization of larger models.

**Usage:**
```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 multi_gpu_example.py

# Multi-node setup (2 nodes, 4 GPUs each)
# Node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<IP> --master_port=29500 multi_gpu_example.py

# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=<IP> --master_port=29500 multi_gpu_example.py
```

**Features:**
- Uses facebook/opt-1.3b (larger model requiring multiple GPUs)
- Distributed calibration data loading
- Coordinated quantization across GPUs
- Memory usage monitoring

### 3. `distributed_example.py` - Comprehensive Example
A full-featured example with command-line arguments and flexible configuration.

**Usage:**
```bash
# Single GPU
python distributed_example.py --model facebook/opt-125m

# Multi-GPU
torchrun --nproc_per_node=4 distributed_example.py --model facebook/opt-1.3b --distributed

# Custom configuration
python distributed_example.py \
    --model microsoft/DialoGPT-medium \
    --bits 8 \
    --group-size 64 \
    --calibration-samples 256 \
    --save-path ./my_quantized_model
```

**Command-line Arguments:**
- `--model`: HuggingFace model name or path
- `--bits`: Quantization bits (2, 3, 4, 8)
- `--group-size`: Group size for quantization
- `--calibration-samples`: Number of calibration samples
- `--seq-len`: Sequence length for calibration
- `--distributed`: Enable distributed mode
- `--save-path`: Path to save quantized model
- `--load-path`: Path to load pre-quantized model

## Requirements

Before running the examples, install the package and its dependencies:

```bash
# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev,cuda]"
```

## Hardware Requirements

### Single GPU Examples
- **Minimum**: 8GB VRAM (for opt-125m)
- **Recommended**: 16GB+ VRAM

### Multi-GPU Examples
- **Minimum**: 2x 16GB VRAM GPUs (for opt-1.3b)
- **Recommended**: 4x 24GB+ VRAM GPUs

## Dataset Requirements

The examples use the WikiText-2 dataset for calibration, which will be automatically downloaded from HuggingFace Hub on first run.

## Performance Tips

1. **Memory Management**: Use `torch.cuda.empty_cache()` between quantization steps if running into memory issues.

2. **Calibration Data**: More calibration samples generally improve quantization quality but increase processing time.

3. **Group Size**: Smaller group sizes (e.g., 64 vs 128) can improve quality but increase memory usage.

4. **Distributed Setup**: Ensure all nodes have the same CUDA version and network connectivity.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **Distributed Timeout**: Check network connectivity between nodes
3. **Model Loading**: Ensure sufficient CPU RAM for model loading

### Debug Mode

Enable debug logging:
```bash
export PYTHONPATH=.
export DISTRIBUTED_GPTQ_LOG_LEVEL=DEBUG
python single_gpu_example.py
```

## Extending the Examples

To adapt these examples for your own models:

1. Replace the model loading code with your model
2. Adjust calibration data preparation for your use case
3. Modify quantization configuration as needed
4. Add custom evaluation metrics if required

For more advanced usage, see the main package documentation.
