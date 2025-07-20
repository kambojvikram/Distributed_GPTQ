"""
Distributed coordination for GPTQ quantization.

This module provides coordination logic for distributed quantization
across multiple GPUs and nodes.
"""

import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
import logging
import contextlib

from .communication import CommunicationManager

logger = logging.getLogger(__name__)


class QuantizationMode(Enum):
    """Quantization modes for distributed processing."""
    SINGLE_GPU = "single_gpu"
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


@dataclass
class DistributedConfig:
    """Configuration for distributed quantization."""
    mode: QuantizationMode = QuantizationMode.SINGLE_GPU
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"
    init_timeout: int = 1800  # 30 minutes
    
    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1
    
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class DistributedGPTQCoordinator:
    """
    Coordinates distributed GPTQ quantization across multiple processes.
    """
    
    def __init__(self, config: DistributedConfig):
        """
        Initialize distributed coordinator.
        
        Args:
            config: Distributed configuration
        """
        self.config = config
        self.device = None
        self.comm_manager = None
        self._is_initialized = False
        
    def initialize(self):
        """Initialize distributed processing."""
        if self._is_initialized:
            return
            
        if self.config.is_distributed:
            # Initialize distributed process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                    rank=self.config.rank,
                    world_size=self.config.world_size,
                    timeout=torch.distributed.distributed_c10d.default_pg_timeout
                )
            
            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
                self.device = torch.device(f"cuda:{self.config.local_rank}")
            else:
                self.device = torch.device("cpu")
                
            # Initialize communication manager
            self.comm_manager = CommunicationManager(
                self.config.rank,
                self.config.world_size,
                self.config.backend
            )
            self.comm_manager.initialize(
                self.config.master_addr,
                self.config.master_port
            )
        else:
            # Single GPU setup
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
                
        self._is_initialized = True
        
        if self.config.is_main_process:
            logger.info(f"Initialized distributed coordinator with {self.config.world_size} processes")
    
    @contextlib.contextmanager
    def distributed_context(self):
        """Context manager for distributed operations."""
        self.initialize()
        try:
            yield
        finally:
            self.cleanup()
    
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare model for distributed quantization.
        
        Args:
            model: Model to prepare
            
        Returns:
            Prepared model
        """
        if not self._is_initialized:
            self.initialize()
            
        # Move model to device
        model = model.to(self.device)
        
        # Wrap with distributed data parallel if needed
        if self.config.mode == QuantizationMode.DATA_PARALLEL and self.config.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank
            )
        
        return model
    
    def distribute_calibration_data(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> torch.utils.data.DataLoader:
        """
        Distribute calibration data across processes.
        
        Args:
            dataloader: Original dataloader
            
        Returns:
            Distributed dataloader
        """
        if not self.config.is_distributed:
            return dataloader
            
        # Create distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataloader.dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=False
        )
        
        # Create new dataloader with distributed sampler
        distributed_loader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory
        )
        
        return distributed_loader
    
    def quantize_layer_distributed(
        self,
        layer: torch.nn.Module,
        layer_inputs: List[torch.Tensor],
        quantization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Quantize a layer in distributed fashion.
        
        Args:
            layer: Layer to quantize
            layer_inputs: Input tensors for the layer
            quantization_config: Quantization configuration
            
        Returns:
            Quantization statistics
        """
        from ..core.gptq import GPTQ
        
        # Initialize GPTQ quantizer
        gptq = GPTQ(
            layer=layer,
            bits=quantization_config['bits'],
            group_size=quantization_config['group_size'],
            actorder=quantization_config.get('actorder', False),
            percdamp=quantization_config.get('percdamp', 0.01),
            blocksize=quantization_config.get('blocksize', 128),
            device=self.device
        )
        
        # Add calibration batches
        for inp in layer_inputs:
            inp = inp.to(self.device)
            with torch.no_grad():
                out = layer(inp)
            gptq.add_batch(inp, out)
        
        # Synchronize Hessian across processes if distributed
        if self.config.is_distributed and self.comm_manager:
            # All-reduce Hessian matrix
            if hasattr(gptq, 'H') and gptq.H is not None:
                self.comm_manager.all_reduce_tensor(gptq.H)
                gptq.H /= self.config.world_size  # Average
        
        # Perform quantization
        stats = gptq.quantize()
        
        # Broadcast quantized weights from rank 0
        if self.config.is_distributed and self.comm_manager:
            for name, param in layer.named_parameters():
                self.comm_manager.broadcast_tensor(param.data, src_rank=0)
        
        return stats
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics."""
        if self.device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated(self.device) / 1e9,
                'reserved_gb': torch.cuda.memory_reserved(self.device) / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / 1e9,
                'max_reserved_gb': torch.cuda.max_memory_reserved(self.device) / 1e9,
            }
        return {}
    
    def barrier(self):
        """Synchronize all processes."""
        if self.config.is_distributed and self.comm_manager:
            self.comm_manager.barrier()
    
    def cleanup(self):
        """Clean up distributed resources."""
        if self.comm_manager:
            self.comm_manager.cleanup()
        
        if self.config.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        self._is_initialized = False


def create_distributed_config(
    mode: Union[str, QuantizationMode] = "auto"
) -> DistributedConfig:
    """
    Create distributed configuration automatically or from environment.
    
    Args:
        mode: Quantization mode or "auto" to detect
        
    Returns:
        Distributed configuration
    """
    import os
    
    # Get values from environment variables (set by torchrun)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    # Determine mode
    if mode == "auto":
        if world_size == 1:
            mode = QuantizationMode.SINGLE_GPU
        else:
            mode = QuantizationMode.DATA_PARALLEL
    elif isinstance(mode, str):
        mode = QuantizationMode(mode)
    
    return DistributedConfig(
        mode=mode,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port
    )
        self.tasks.put(task)

    def get_result(self):
        return self.results.get()

    def stop_workers(self):
        for _ in range(self.num_workers):
            self.tasks.put(None)  # Send exit signal to workers
        for worker in self.workers:
            worker.join()  # Wait for workers to finish

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    coordinator = Coordinator(num_workers=4)
    coordinator.start_workers()

    for i in range(10):
        coordinator.add_task(f"Task {i}")

    for _ in range(10):
        result = coordinator.get_result()
        logging.info(f"Received: {result}")

    coordinator.stop_workers()