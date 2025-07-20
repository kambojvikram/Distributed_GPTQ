"""
Multi-GPU GPTQ Quantization Example

This example demonstrates how to quantize a model using GPTQ across multiple GPUs.

Usage:
    torchrun --nproc_per_node=4 multi_gpu_example.py
    
Requirements:
    - Multiple GPUs
    - torch.distributed
"""

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
import os

from distributed_gptq.core.quantizer import DistributedGPTQuantizer, QuantizationConfig
from distributed_gptq.distributed.coordinator import create_distributed_config, QuantizationMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed environment."""
    if not dist.is_initialized():
        # Check if we're running with torchrun
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            # Initialize process group
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size
            )
            
            # Set CUDA device
            torch.cuda.set_device(local_rank)
            
            return rank, world_size, local_rank
        else:
            logger.warning("Not running with torchrun. Using single GPU mode.")
            return 0, 1, 0
    else:
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))


def get_calibration_data(tokenizer, n_samples=128, seq_len=1024):
    """Get calibration data from WikiText dataset."""
    # Only load on rank 0 to avoid duplicate downloads
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0:
        logger.info("Loading calibration dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        
        samples = []
        for i in range(min(n_samples, len(dataset))):
            text = dataset[i]["text"]
            if len(text) > 20:  # Skip very short texts
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=seq_len,
                    padding="max_length",
                    return_tensors="pt"
                )
                samples.append(tokens.input_ids)
        
        calibration_data = torch.cat(samples[:n_samples], dim=0)
        logger.info(f"Prepared {calibration_data.shape[0]} calibration samples")
    else:
        calibration_data = None
    
    # Broadcast calibration data to all processes
    if dist.is_initialized():
        if rank == 0:
            # Get tensor info to broadcast
            tensor_shape = torch.tensor(calibration_data.shape, dtype=torch.long)
        else:
            tensor_shape = torch.tensor([0, 0], dtype=torch.long)
        
        # Broadcast shape
        dist.broadcast(tensor_shape, src=0)
        
        # Create tensor on non-rank-0 processes
        if rank != 0:
            calibration_data = torch.zeros(tensor_shape.tolist(), dtype=torch.long)
        
        # Broadcast data
        dist.broadcast(calibration_data, src=0)
    
    return calibration_data


def main():
    """Main function for multi-GPU quantization example."""
    
    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        logger.info(f"Starting multi-GPU quantization with {world_size} processes")
    
    # Model configuration
    model_name = "facebook/opt-1.3b"  # Larger model for multi-GPU
    
    if rank == 0:
        logger.info(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to local GPU
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")
    
    # Get calibration data
    calibration_data = get_calibration_data(tokenizer, n_samples=64, seq_len=512)
    
    # Setup distributed quantization config
    dist_config = create_distributed_config(
        mode=QuantizationMode.MULTI_GPU,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank
    )
    
    # Setup quantization configuration
    quant_config = QuantizationConfig(
        bits=4,
        group_size=128,
        calibration_samples=64,
        actorder=False,
        percdamp=0.01
    )
    
    # Create distributed quantizer
    quantizer = DistributedGPTQuantizer(
        quantization_config=quant_config,
        distributed_config=dist_config
    )
    
    # Progress callback (only on rank 0)
    def progress_callback(current, total, layer_name, stats):
        if rank == 0:
            error = stats.get('total_error', 0)
            logger.info(f"Progress: {current}/{total} - {layer_name} - Error: {error:.4f}")
    
    # Quantize the model
    if rank == 0:
        logger.info("Starting distributed quantization...")
    
    quantized_model = quantizer.quantize_model(
        model,
        calibration_data,
        save_path=f"./quantized_opt_1.3b" if rank == 0 else None,
        progress_callback=progress_callback if rank == 0 else None
    )
    
    if rank == 0:
        logger.info("Distributed quantization completed!")
    
    # Test the quantized model (only on rank 0)
    if rank == 0:
        logger.info("Testing quantized model...")
        test_text = "The future of distributed computing is"
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        
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
        
        # Memory usage across GPUs
        logger.info("\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            logger.info("Multi-GPU example completed successfully!")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()