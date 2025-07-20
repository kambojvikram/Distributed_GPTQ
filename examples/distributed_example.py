"""
Example usage of distributed-gptq package.

Single GPU:
    python example.py --model facebook/opt-125m

Multi-GPU (single node):
    torchrun --nproc_per_node=4 example.py --model facebook/opt-1.3b --distributed

Multi-node (2 nodes, 4 GPUs each):
    # On node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<IP> --master_port=29500 example.py --model facebook/opt-6.7b --distributed
    
    # On node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=<IP> --master_port=29500 example.py --model facebook/opt-6.7b --distributed
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
from tqdm import tqdm

from distributed_gptq import (
    DistributedGPTQuantizer,
    QuantizationConfig,
    create_distributed_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_calibration_data(tokenizer, n_samples=128, seq_len=2048):
    """Get calibration data from a dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    samples = []
    for i in range(n_samples):
        text = dataset[i]["text"]
        if len(text) > 10:  # Skip very short texts
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="pt"
            )
            samples.append(tokens.input_ids)
    
    return torch.cat(samples, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Distributed GPTQ Quantization Example")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="Model to quantize")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for quantization")
    parser.add_argument("--calibration-samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length for calibration")
    parser.add_argument("--distributed", action="store_true",
                        help="Use distributed quantization")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save quantized model")
    parser.add_argument("--load-path", type=str, default=None,
                        help="Path to load quantized model from")
    
    args = parser.parse_args()
    
    # Setup distributed config
    if args.distributed:
        dist_config = create_distributed_config(mode="auto")
    else:
        dist_config = create_distributed_config(mode="single_gpu")
    
    # Only print on main process
    is_main = dist_config.rank == 0
    
    if is_main:
        logger.info(f"Quantizing model: {args.model}")
        logger.info(f"Mode: {dist_config.mode.value}")
        logger.info(f"World size: {dist_config.world_size}")
    
    # Load model and tokenizer
    if is_main:
        logger.info("Loading model and tokenizer...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto" if dist_config.mode.value == "single_gpu" else None
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup quantization config
    quant_config = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        calibration_samples=args.calibration_samples,
    )
    
    # Create quantizer
    quantizer = DistributedGPTQuantizer(
        quantization_config=quant_config,
        distributed_config=dist_config
    )
    
    if args.load_path:
        # Load pre-quantized model
        if is_main:
            logger.info(f"Loading quantized model from {args.load_path}")
        model = quantizer.load_quantized_model(model, args.load_path)
    else:
        # Get calibration data
        if is_main:
            logger.info("Preparing calibration data...")
        calibration_data = get_calibration_data(
            tokenizer,
            n_samples=args.calibration_samples,
            seq_len=args.seq_len
        )
        
        # Progress callback
        def progress_callback(current, total, layer_name, stats):
            logger.info(
                f"Progress: {current}/{total} - Layer: {layer_name} - "
                f"Error: {stats.get('total_error', 0):.4f}"
            )
        
        # Quantize model
        if is_main:
            logger.info("Starting quantization...")
        
        model = quantizer.quantize_model(
            model,
            calibration_data,
            save_path=args.save_path,
            progress_callback=progress_callback if is_main else None
        )
        
        if is_main:
            logger.info("Quantization complete!")
    
    # Test the quantized model
    if is_main:
        logger.info("\nTesting quantized model...")
        test_text = "The quick brown fox"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        
        # Print memory usage
        if torch.cuda.is_available():
            logger.info(f"\nGPU Memory Usage:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                logger.info(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


if __name__ == "__main__":
    main()
