"""
Command-line interface for distributed-gptq.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
import torch
from typing import Optional
import os
import torch.multiprocessing as mp
import torch.distributed as dist

from .utils.logging_utils import setup_logging, get_logger
from .utils.gpu_utils import check_gpu_availability, print_gpu_info
from .utils.data_utils import load_common_datasets, create_calibration_dataloader
from .core.quantizer import QuantizationConfig
from .core.gptq import GPTQQuantizer
from .models.transformers_model import load_transformers_model
from .distributed.coordinator import DistributedCoordinator
from .utils.benchmark import PerformanceMonitor
from .utils.conversion import convert_model_format
from safetensors import save_file


from . import (
    __version__
)

# Remove the basicConfig and get a logger instance
logger = get_logger(__name__)


def quantize_command(args):
    """Handle quantize command."""
    if args.distributed:
        mp.spawn(
            distributed_quantize,
            args=(args,),
            nprocs=args.world_size,
            join=True
        )
    else:
        single_gpu_quantize(args)


def benchmark_command(args):
    """Handle benchmark command."""
    logger.info("Running benchmark...")
    
    # Import benchmark utilities
    from .utils.benchmark import run_benchmark
    
    results = run_benchmark(
        model_path=args.model,
        quantized_path=args.quantized_model,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Print results
    print("\nBenchmark Results:")
    print("-" * 50)
    print(f"Model: {args.model}")
    print(f"Quantized Model: {args.quantized_model}")
    print(f"Compression Ratio: {results['compression_ratio']:.2f}x")
    print(f"Speed Up: {results['speedup']:.2f}x")
    print(f"Memory Reduction: {results['memory_reduction']:.2f}x")
    print(f"Perplexity Change: {results['perplexity_change']:.4f}")
    print("-" * 50)
    
    # Save results if requested
    if args.save_results:
        results_path = Path(args.save_results)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")


def convert_command(args):
    """Handle convert command."""
    logger.info(f"Converting {args.input} to {args.output_format}")
    
    convert_model_format(
        input_path=args.input,
        output_path=args.output,
        input_format=args.input_format,
        output_format=args.output_format,
        optimize=args.optimize
    )
    
    logger.info(f"Conversion complete! Output saved to {args.output}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    # Setup logging
    log_level = args.log_level if hasattr(args, 'log_level') else 'INFO'
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Distributed GPTQ Quantization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"distributed-gptq {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Quantize command parser
    quantize_parser = subparsers.add_parser("quantize", help="Quantize a model")
    add_quantize_arguments(quantize_parser)
    quantize_parser.set_defaults(func=quantize_command)

    # Benchmark command parser
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark a quantized model")
    add_benchmark_arguments(bench_parser)
    bench_parser.set_defaults(func=benchmark_command)

    # Convert command parser
    convert_parser = subparsers.add_parser("convert", help="Convert model format")
    add_convert_arguments(convert_parser)
    convert_parser.set_defaults(func=convert_command)

    return parser


def add_quantize_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the quantize command."""
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Model name or path (HuggingFace model identifier)'
    )
    model_group.add_argument(
        '--model-type',
        type=str,
        choices=['transformers', 'custom'],
        default='transformers',
        help='Type of model to quantize'
    )
    model_group.add_argument(
        '--torch-dtype',
        type=str,
        choices=['float16', 'bfloat16', 'float32'],
        default='float16',
        help='Data type for model weights'
    )
    
    # Quantization arguments
    quant_group = parser.add_argument_group('Quantization Configuration')
    quant_group.add_argument(
        '--bits',
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help='Number of bits for quantization'
    )
    quant_group.add_argument(
        '--group-size',
        type=int,
        default=128,
        help='Group size for quantization'
    )
    quant_group.add_argument(
        '--act-order',
        action='store_true',
        help='Use activation order for quantization'
    )
    quant_group.add_argument(
        '--true-sequential',
        action='store_true',
        help='Use true sequential quantization'
    )
    quant_group.add_argument(
        '--damp',
        type=float,
        default=0.01,
        help='Damping factor for Hessian'
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Calibration Data')
    data_group.add_argument(
        '--dataset',
        type=str,
        default='wikitext',
        choices=['wikitext', 'c4', 'ptb', 'custom'],
        help='Calibration dataset to use'
    )
    data_group.add_argument(
        '--dataset-path',
        type=str,
        help='Path to custom dataset file'
    )
    data_group.add_argument(
        '--num-samples',
        type=int,
        default=128,
        help='Number of calibration samples'
    )
    data_group.add_argument(
        '--seq-len',
        type=int,
        default=2048,
        help='Sequence length for calibration'
    )
    data_group.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for calibration'
    )
    
    # Distributed arguments
    dist_group = parser.add_argument_group('Distributed Configuration')
    dist_group.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed quantization'
    )
    dist_group.add_argument(
        '--world-size',
        type=int,
        default=1,
        help='Number of processes for distributed training'
    )
    dist_group.add_argument(
        '--rank',
        type=int,
        default=0,
        help='Rank of current process'
    )
    dist_group.add_argument(
        '--master-addr',
        type=str,
        default='localhost',
        help='Master node address'
    )
    dist_group.add_argument(
        '--master-port',
        type=str,
        default='12355',
        help='Master node port'
    )
    dist_group.add_argument(
        '--backend',
        type=str,
        default='nccl',
        choices=['nccl', 'gloo', 'mpi'],
        help='Distributed backend'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='./quantized_models',
        help='Output directory for quantized model'
    )
    output_group.add_argument(
        '--save-format',
        type=str,
        choices=['pytorch', 'safetensors', 'both'],
        default='safetensors',
        help='Format to save quantized model'
    )
    output_group.add_argument(
        '--log-dir',
        type=str,
        help='Directory for log files'
    )
    output_group.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    # Performance arguments
    perf_group = parser.add_argument_group('Performance Configuration')
    perf_group.add_argument(
        '--eval-ppl',
        action='store_true',
        help='Evaluate perplexity after quantization'
    )
    perf_group.add_argument(
        '--eval-dataset',
        type=str,
        default='wikitext',
        help='Dataset for perplexity evaluation'
    )
    perf_group.add_argument(
        '--benchmark',
        action='store_true',
        help='Run throughput benchmark'
    )
    perf_group.add_argument(
        '--profile',
        action='store_true',
        help='Enable detailed profiling'
    )


def add_benchmark_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the benchmark command."""
    parser.add_argument(
        "model",
        help="Original model path"
    )
    parser.add_argument(
        "quantized_model",
        help="Quantized model path"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=100,
        help="Number of test samples (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for testing (default: 1)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run benchmark on (default: cuda)"
    )
    parser.add_argument(
        "--save-results",
        help="Path to save benchmark results"
    )


def add_convert_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the convert command."""
    parser.add_argument(
        "input",
        help="Input model path"
    )
    parser.add_argument(
        "output",
        help="Output model path"
    )
    parser.add_argument(
        "--input-format",
        choices=["pytorch", "safetensors", "gguf"],
        required=True,
        help="Input format"
    )
    parser.add_argument(
        "--output-format",
        choices=["pytorch", "safetensors", "gguf"],
        required=True,
        help="Output format"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize during conversion"
    )


def load_model_and_tokenizer(args):
    """Load model and tokenizer based on arguments."""
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    
    if args.model_type == 'transformers':
        model_wrapper = load_transformers_model(
            args.model,
            torch_dtype=dtype_map[args.torch_dtype],
            device_map='auto' if args.distributed else None
        )
        return model_wrapper.model, model_wrapper.tokenizer
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")


def prepare_calibration_data(args, tokenizer):
    """Prepare calibration data based on arguments."""
    if args.dataset == 'custom' and args.dataset_path:
        # Load custom dataset
        calibration_texts = []
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                calibration_texts.append(line.strip())
    else:
        # Load common dataset
        calibration_texts = load_common_datasets(
            args.dataset,
            split='train',
            num_samples=args.num_samples
        )
    
    # Create dataloader
    dataloader = create_calibration_dataloader(
        calibration_texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.seq_len,
        num_samples=args.num_samples
    )
    
    return dataloader


def single_gpu_quantize(args):
    """Run quantization on a single GPU."""
    # Setup logging
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = create_log_directory()
    
    logger = setup_logging(
        log_level=args.log_level,
        log_file=log_dir / "quantization.log",
        rank=0,
        world_size=1
    )
    
    logger.info("Starting single GPU quantization")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check GPU availability
    if not check_gpu_availability():
        logger.error("No GPU available for quantization")
        sys.exit(1)
    
    print_gpu_info()
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args)
    device = torch.device('cuda:0')
    model = model.to(device)
    
    # Prepare calibration data
    logger.info("Preparing calibration data")
    calibration_dataloader = prepare_calibration_data(args, tokenizer)
    
    # Create quantization config
    config = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        act_order=args.act_order,
        true_sequential=args.true_sequential,
        damp=args.damp
    )
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    # Initialize quantizer
    quantizer = GPTQQuantizer(config=config)
    
    # Run quantization
    logger.info("Starting quantization process")
    monitor.start_timer('total_quantization')
    
    quantized_model = quantizer.quantize(
        model=model,
        calibration_data=calibration_dataloader,
        device=device
    )
    
    total_time = monitor.end_timer('total_quantization')
    monitor.metrics.total_time = total_time
    
    # Record metrics
    monitor.record_model_metrics(model, quantized_model, args.bits)
    
    # Evaluate perplexity if requested
    if args.eval_ppl:
        logger.info("Evaluating perplexity")
        eval_dataloader = prepare_calibration_data(args, tokenizer)
        monitor.record_perplexity(model, quantized_model, eval_dataloader, device)
    
    # Benchmark if requested
    if args.benchmark:
        logger.info("Running benchmark")
        bench_dataloader = prepare_calibration_data(args, tokenizer)
        throughput = monitor.measure_throughput(quantized_model, bench_dataloader, device)
        monitor.metrics.tokens_per_second = throughput['tokens_per_second']
        monitor.metrics.samples_per_second = throughput['samples_per_second']
    
    # Save quantized model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_output_dir = output_dir / f"{Path(args.model).name}_quantized_{args.bits}bit"
    model_output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Saving quantized model to {model_output_dir}")
    
    if args.save_format in ['pytorch', 'both']:
        torch.save(quantized_model.state_dict(), model_output_dir / "pytorch_model.bin")
    
    if args.save_format in ['safetensors', 'both']:
        try:
            save_file(quantized_model.state_dict(), model_output_dir / "model.safetensors")
        except ImportError:
            logger.warning("safetensors not available, using pytorch format")
            torch.save(quantized_model.state_dict(), model_output_dir / "pytorch_model.bin")
    
    # Save configuration
    config_dict = {
        'quantization_config': config.__dict__,
        'model_config': vars(args)
    }
    with open(model_output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save metrics
    monitor.finalize_metrics()
    monitor.metrics.save(log_dir / "metrics.json")
    monitor.print_summary()
    
    logger.info("Quantization completed successfully")


def distributed_quantize(rank: int, args):
    """Run distributed quantization."""
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    
    # Initialize process group
    dist.init_process_group(
        backend=args.backend,
        rank=rank,
        world_size=args.world_size
    )
    
    # Setup logging
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = create_log_directory()
    
    logger = setup_logging(
        log_level=args.log_level,
        log_file=log_dir / f"quantization_rank_{rank}.log",
        rank=rank,
        world_size=args.world_size
    )
    
    logger.info(f"Starting distributed quantization on rank {rank}")
    
    try:
        # Set device for this process
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        
        # Load model and tokenizer
        logger.info(f"Loading model on rank {rank}")
        model, tokenizer = load_model_and_tokenizer(args)
        model = model.to(device)
        
        # Prepare calibration data
        logger.info("Preparing calibration data")
        calibration_dataloader = prepare_calibration_data(args, tokenizer)
        
        # Create quantization config
        config = QuantizationConfig(
            bits=args.bits,
            group_size=args.group_size,
            act_order=args.act_order,
            true_sequential=args.true_sequential,
            damp=args.damp
        )
        
        # Initialize distributed coordinator
        coordinator = DistributedCoordinator(
            rank=rank,
            world_size=args.world_size,
            config=config
        )
        
        # Run distributed quantization
        logger.info("Starting distributed quantization")
        quantized_model = coordinator.quantize(
            model=model,
            calibration_data=calibration_dataloader,
            device=device
        )
        
        # Save model on rank 0
        if rank == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_output_dir = output_dir / f"{Path(args.model).name}_quantized_{args.bits}bit"
            model_output_dir.mkdir(exist_ok=True)
            
            logger.info(f"Saving quantized model to {model_output_dir}")
            
            if args.save_format in ['pytorch', 'both']:
                torch.save(quantized_model.state_dict(), model_output_dir / "pytorch_model.bin")
            
            if args.save_format in ['safetensors', 'both']:
                try:
                    save_file(quantized_model.state_dict(), model_output_dir / "model.safetensors")
                except ImportError:
                    logger.warning("safetensors not available, using pytorch format")
                    torch.save(quantized_model.state_dict(), model_output_dir / "pytorch_model.bin")
            
            # Save configuration
            config_dict = {
                'quantization_config': config.__dict__,
                'model_config': vars(args)
            }
            with open(model_output_dir / "config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Rank {rank} completed quantization")
        
    except Exception as e:
        logger.error(f"Error on rank {rank}: {e}")
        raise
    finally:
        # Cleanup
        dist.destroy_process_group()


def create_log_directory() -> Path:
    """Create a directory for logging."""
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir
