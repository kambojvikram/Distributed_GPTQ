"""
Single GPU GPTQ Quantization Example

This example demonstrates how to quantize a model using GPTQ on a single GPU.

Usage:
    python single_gpu_example.py
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from distributed_gptq.core.quantizer import DistributedGPTQuantizer, QuantizationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(tokenizer, batch_size=4, seq_len=512):
    """Create sample calibration data."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world of technology.",
        "Machine learning models require careful optimization for deployment.",
        "Quantization reduces model size while maintaining performance.",
    ]
    
    inputs = []
    for text in sample_texts[:batch_size]:
        tokens = tokenizer(
            text,
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs.append(tokens.input_ids)
    
    return torch.cat(inputs, dim=0)


def main():
    """Main function for single GPU quantization example."""
    
    # Model configuration
    model_name = "facebook/opt-125m"  # Small model for demo
    
    logger.info(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded with {total_params:,} parameters")
    
    # Create calibration data
    logger.info("Creating calibration data...")
    calibration_data = create_sample_data(tokenizer, batch_size=8, seq_len=256)
    
    # Setup quantization configuration
    quant_config = QuantizationConfig(
        bits=4,
        group_size=128,
        calibration_samples=8,
        actorder=False,
        percdamp=0.01
    )
    
    # Create quantizer (will automatically use single GPU mode)
    quantizer = DistributedGPTQuantizer(quantization_config=quant_config)
    
    # Quantize the model
    logger.info("Starting quantization...")
    quantized_model = quantizer.quantize_model(
        model,
        calibration_data,
        save_path="./quantized_opt_125m"
    )
    
    logger.info("Quantization completed!")
    
    # Test the quantized model
    logger.info("Testing quantized model...")
    test_text = "The future of AI is"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    # Move to same device as model
    device = next(quantized_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate text
    with torch.no_grad():
        outputs = quantized_model.generate(
            **inputs,
            max_length=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")
    
    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory allocated: {allocated:.2f} GB")
    
    logger.info("Single GPU example completed successfully!")


if __name__ == "__main__":
    main()