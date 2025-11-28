"""
Logging utilities for distributed GPTQ.

This module provides logging configuration and utilities for
the distributed quantization process.
"""

import logging
import sys
import os
from typing import Optional, Union, Dict, Any
from pathlib import Path
import time
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for better readability.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


class DistributedLogger:
    """
    Logger for distributed processes with rank-aware formatting.
    """
    
    def __init__(
        self,
        name: str,
        rank: int = 0,
        world_size: int = 1,
        log_level: Union[str, int] = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        use_colors: bool = True
    ):
        """
        Initialize distributed logger.
        
        Args:
            name: Logger name
            rank: Process rank
            world_size: Total number of processes
            log_level: Logging level
            log_file: Optional log file path
            use_colors: Whether to use colored output
        """
        self.rank = rank
        self.world_size = world_size
        self.logger = logging.getLogger(f"{name}_rank_{rank}")
        
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(log_level)
        
        # Create formatters
        if use_colors and sys.stdout.isatty():
            formatter_class = ColoredFormatter
        else:
            formatter_class = logging.Formatter
        
        console_format = f"[Rank {rank}/{world_size}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_format = f"[Rank {rank}/{world_size}] %(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter_class(console_format))
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add rank to filename
            stem = log_file.stem
            suffix = log_file.suffix
            rank_log_file = log_file.parent / f"{stem}_rank_{rank}{suffix}"
            
            file_handler = logging.FileHandler(rank_log_file)
            file_handler.setFormatter(logging.Formatter(file_format))
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)
    
    def log_only_rank0(self, level: int, msg: str, *args, **kwargs):
        """Log message only on rank 0."""
        if self.rank == 0:
            self.logger.log(level, msg, *args, **kwargs)
    
    def info_rank0(self, msg: str, *args, **kwargs):
        """Log info message only on rank 0."""
        self.log_only_rank0(logging.INFO, msg, *args, **kwargs)
    
    def warning_rank0(self, msg: str, *args, **kwargs):
        """Log warning message only on rank 0."""
        self.log_only_rank0(logging.WARNING, msg, *args, **kwargs)


class ProgressLogger:
    """
    Logger for tracking progress during quantization.
    """
    
    def __init__(self, logger: logging.Logger, total_steps: int):
        """
        Initialize progress logger.
        
        Args:
            logger: Base logger
            total_steps: Total number of steps
        """
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
    
    def step(self, description: str = ""):
        """
        Log a step in the process.
        
        Args:
            description: Optional step description
        """
        self.current_step += 1
        current_time = time.time()
        
        # Calculate timing
        elapsed = current_time - self.start_time
        if len(self.step_times) > 0:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_step_time
        else:
            eta = 0
        
        # Log progress
        progress_pct = (self.current_step / self.total_steps) * 100
        msg = f"Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%)"
        
        if description:
            msg += f" - {description}"
        
        if eta > 0:
            eta_str = self._format_time(eta)
            msg += f" - ETA: {eta_str}"
        
        elapsed_str = self._format_time(elapsed)
        msg += f" - Elapsed: {elapsed_str}"
        
        self.logger.info(msg)
        
        # Track step time
        if len(self.step_times) > 0:
            step_time = current_time - self.last_step_time
            self.step_times.append(step_time)
            
            # Keep only recent step times for better ETA estimation
            if len(self.step_times) > 10:
                self.step_times.pop(0)
        
        self.last_step_time = current_time
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


def setup_logging(
    log_level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rank: int = 0,
    world_size: int = 1,
    use_colors: bool = True
) -> DistributedLogger:
    """
    Setup logging for distributed GPTQ.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        rank: Process rank
        world_size: Total number of processes
        use_colors: Whether to use colored output
        
    Returns:
        Configured distributed logger
    """
    # Convert string log level to int
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    return DistributedLogger(
        name="distributed_gptq",
        rank=rank,
        world_size=world_size,
        log_level=log_level,
        log_file=log_file,
        use_colors=use_colors
    )


def log_model_info(logger: logging.Logger, model, model_name: str = "Model"):
    """
    Log information about a model.
    
    Args:
        logger: Logger instance
        model: PyTorch model
        model_name: Name for the model
    """
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory
        memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        logger.info(f"{model_name} Information:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Memory footprint: {memory_mb:.1f}MB")
        
        # Log layer types
        layer_counts = {}
        for module in model.modules():
            layer_type = type(module).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        logger.info(f"  Layer breakdown:")
        for layer_type, count in sorted(layer_counts.items()):
            if count > 1:
                logger.info(f"    {layer_type}: {count}")
            
    except Exception as e:
        logger.warning(f"Could not log model info: {e}")


def log_system_info(logger: logging.Logger):
    """
    Log system information.
    
    Args:
        logger: Logger instance
    """
    import torch
    import platform
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python version: {platform.python_version()}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA available: True")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            logger.info(f"    GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    else:
        logger.info(f"  CUDA available: False")


class Timer:
    """
    Context manager for timing operations.
    """
    
    def __init__(self, logger: logging.Logger, operation_name: str, log_level: int = logging.INFO):
        """
        Initialize timer.
        
        Args:
            logger: Logger instance
            operation_name: Name of the operation being timed
            log_level: Log level for timing messages
        """
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.log(self.log_level, f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            if duration < 60:
                duration_str = f"{duration:.2f}s"
            elif duration < 3600:
                duration_str = f"{duration/60:.1f}m"
            else:
                duration_str = f"{duration/3600:.1f}h"
            
            self.logger.log(self.log_level, f"Completed {self.operation_name} in {duration_str}")


def create_log_directory(base_dir: Union[str, Path] = "logs") -> Path:
    """
    Create a timestamped log directory.
    
    Args:
        base_dir: Base directory for logs
        
    Returns:
        Path to created log directory
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = base_dir / f"gptq_quantization_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
