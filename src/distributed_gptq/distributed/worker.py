"""
Distributed worker for GPTQ quantization.

This module implements worker processes for distributed GPTQ quantization,
handling layer-specific quantization tasks across multiple GPUs/nodes.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from ..core.gptq import GPTQ
from ..core.quantizer import QuantizationConfig
from ..utils.gpu_utils import GPUMemoryManager
from ..utils.logging_utils import get_logger
from .communication import DistributedCommunicator


logger = get_logger(__name__)


@dataclass
class QuantizationTask:
    """
    Task for quantizing a specific layer.
    """
    layer_name: str
    layer_state_dict: Dict[str, torch.Tensor]
    layer_config: Dict[str, Any]
    calibration_data: List[torch.Tensor]
    task_id: str
    dependencies: List[str] = None


@dataclass
class QuantizationResult:
    """
    Result of layer quantization.
    """
    layer_name: str
    quantized_state_dict: Dict[str, torch.Tensor]
    quantization_stats: Dict[str, float]
    task_id: str
    success: bool
    error_message: Optional[str] = None


class DistributedGPTQWorker:
    """
    Worker process for distributed GPTQ quantization.
    
    Each worker is responsible for quantizing assigned layers
    and communicating results back to the coordinator.
    """
    
    def __init__(
        self,
        worker_id: int,
        device: torch.device,
        config: QuantizationConfig,
        world_size: int,
        rank: int
    ):
        """
        Initialize the distributed worker.
        
        Args:
            worker_id: Unique worker identifier
            device: Device for this worker
            config: Quantization configuration
            world_size: Total number of processes
            rank: Process rank
        """
        self.worker_id = worker_id
        self.device = device
        self.config = config
        self.world_size = world_size
        self.rank = rank
        
        # Initialize components
        self.memory_manager = GPUMemoryManager(device)
        self.communicator = DistributedCommunicator(rank, world_size)
        
        # Task management
        self.current_task: Optional[QuantizationTask] = None
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        # Statistics
        self.start_time = time.time()
        self.total_layers_processed = 0
        self.total_processing_time = 0.0
        
        logger.info(f"Initialized worker {worker_id} on device {device} (rank {rank})")
    
    def run(self):
        """
        Main worker loop.
        
        Continuously receives tasks from coordinator, processes them,
        and sends results back.
        """
        logger.info(f"Worker {self.worker_id} starting main loop")
        
        try:
            while True:
                # Receive task from coordinator
                task = self._receive_task()
                
                if task is None:
                    logger.info(f"Worker {self.worker_id} received shutdown signal")
                    break
                
                # Process the task
                result = self._process_task(task)
                
                # Send result back to coordinator
                self._send_result(result)
                
        except Exception as e:
            logger.error(f"Worker {self.worker_id} encountered error: {e}")
            logger.error(traceback.format_exc())
        finally:
            self._cleanup()
    
    def _receive_task(self) -> Optional[QuantizationTask]:
        """
        Receive a quantization task from the coordinator.
        
        Returns:
            QuantizationTask or None if shutdown signal received
        """
        try:
            # Use distributed communication to receive task
            task_data = self.communicator.broadcast_object(None, src=0)
            
            if task_data is None or task_data.get('type') == 'shutdown':
                return None
            
            # Reconstruct task from received data
            task = QuantizationTask(
                layer_name=task_data['layer_name'],
                layer_state_dict=task_data['layer_state_dict'],
                layer_config=task_data['layer_config'],
                calibration_data=task_data['calibration_data'],
                task_id=task_data['task_id'],
                dependencies=task_data.get('dependencies', [])
            )
            
            logger.info(f"Worker {self.worker_id} received task: {task.layer_name}")
            return task
            
        except Exception as e:
            logger.error(f"Error receiving task: {e}")
            return None
    
    def _process_task(self, task: QuantizationTask) -> QuantizationResult:
        """
        Process a quantization task.
        
        Args:
            task: Task to process
            
        Returns:
            QuantizationResult with outcomes
        """
        self.current_task = task
        start_time = time.time()
        
        logger.info(f"Worker {self.worker_id} processing layer: {task.layer_name}")
        
        try:
            # Clear GPU memory before processing
            self.memory_manager.clear_cache()
            
            # Create layer from state dict
            layer = self._reconstruct_layer(task.layer_state_dict, task.layer_config)
            layer = layer.to(self.device)
            
            # Initialize GPTQ for this layer
            gptq = GPTQ(
                layer=layer,
                layer_name=task.layer_name,
                bits=self.config.bits,
                group_size=self.config.group_size,
                actorder=self.config.actorder,
                percdamp=self.config.percdamp,
                device=self.device
            )
            
            # Add calibration data
            for batch in task.calibration_data:
                batch = batch.to(self.device)
                gptq.add_batch(batch)
            
            # Perform quantization
            quantized_weight = gptq.quantize()
            
            # Get quantization statistics
            stats = {
                'compression_ratio': self._calculate_compression_ratio(layer.weight, quantized_weight),
                'quantization_error': self._calculate_quantization_error(layer.weight, quantized_weight),
                'processing_time': time.time() - start_time,
                'memory_used_mb': self.memory_manager.get_memory_info()['allocated']
            }
            
            # Update layer state dict with quantized weights
            quantized_state_dict = task.layer_state_dict.copy()
            quantized_state_dict['weight'] = quantized_weight
            
            # Create successful result
            result = QuantizationResult(
                layer_name=task.layer_name,
                quantized_state_dict=quantized_state_dict,
                quantization_stats=stats,
                task_id=task.task_id,
                success=True
            )
            
            self.completed_tasks.append(task.task_id)
            self.total_layers_processed += 1
            self.total_processing_time += stats['processing_time']
            
            logger.info(f"Worker {self.worker_id} completed layer {task.layer_name} "
                       f"in {stats['processing_time']:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error quantizing layer {task.layer_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Create failed result
            result = QuantizationResult(
                layer_name=task.layer_name,
                quantized_state_dict={},
                quantization_stats={'processing_time': time.time() - start_time},
                task_id=task.task_id,
                success=False,
                error_message=error_msg
            )
            
            self.failed_tasks.append(task.task_id)
            return result
        
        finally:
            self.current_task = None
            # Clean up GPU memory
            self.memory_manager.clear_cache()
    
    def _reconstruct_layer(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> nn.Module:
        """
        Reconstruct a layer from its state dict and configuration.
        
        Args:
            state_dict: Layer state dictionary
            config: Layer configuration
            
        Returns:
            Reconstructed layer
        """
        layer_type = config.get('type', 'Linear')
        
        if layer_type == 'Linear':
            in_features = config['in_features']
            out_features = config['out_features']
            bias = config.get('bias', True)
            
            layer = nn.Linear(in_features, out_features, bias=bias)
            layer.load_state_dict(state_dict)
            return layer
        
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    
    def _calculate_compression_ratio(self, original: torch.Tensor, quantized: torch.Tensor) -> float:
        """Calculate compression ratio between original and quantized weights."""
        original_bits = original.element_size() * 8
        quantized_bits = self.config.bits
        return original_bits / quantized_bits
    
    def _calculate_quantization_error(self, original: torch.Tensor, quantized: torch.Tensor) -> float:
        """Calculate quantization error (MSE)."""
        return torch.mean((original - quantized) ** 2).item()
    
    def _send_result(self, result: QuantizationResult):
        """
        Send quantization result back to coordinator.
        
        Args:
            result: Result to send
        """
        try:
            # Convert result to dictionary for transmission
            result_data = {
                'type': 'result',
                'layer_name': result.layer_name,
                'quantized_state_dict': result.quantized_state_dict,
                'quantization_stats': result.quantization_stats,
                'task_id': result.task_id,
                'success': result.success,
                'error_message': result.error_message,
                'worker_id': self.worker_id
            }
            
            self.communicator.gather_objects(result_data, dst=0)
            
            logger.info(f"Worker {self.worker_id} sent result for {result.layer_name}")
            
        except Exception as e:
            logger.error(f"Error sending result: {e}")
    
    def _cleanup(self):
        """Clean up worker resources."""
        logger.info(f"Worker {self.worker_id} cleaning up")
        
        # Log final statistics
        total_time = time.time() - self.start_time
        avg_processing_time = (self.total_processing_time / self.total_layers_processed 
                              if self.total_layers_processed > 0 else 0)
        
        logger.info(f"Worker {self.worker_id} statistics:")
        logger.info(f"  Total runtime: {total_time:.2f}s")
        logger.info(f"  Layers processed: {self.total_layers_processed}")
        logger.info(f"  Average processing time: {avg_processing_time:.2f}s/layer")
        logger.info(f"  Failed tasks: {len(self.failed_tasks)}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def start_worker_process(
    worker_id: int,
    device: str,
    config: QuantizationConfig,
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: str
):
    """
    Start a worker process for distributed quantization.
    
    Args:
        worker_id: Worker identifier
        device: Device string (e.g., 'cuda:0')
        config: Quantization configuration
        world_size: Total number of processes
        rank: Process rank
        master_addr: Master node address
        master_port: Master node port
    """
    try:
        # Initialize distributed environment
        import os
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        dist.init_process_group(
            backend='nccl' if 'cuda' in device else 'gloo',
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        if 'cuda' in device:
            torch.cuda.set_device(device)
        
        device_obj = torch.device(device)
        
        # Create and run worker
        worker = DistributedGPTQWorker(
            worker_id=worker_id,
            device=device_obj,
            config=config,
            world_size=world_size,
            rank=rank
        )
        
        worker.run()
        
    except Exception as e:
        logger.error(f"Worker process {worker_id} failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class LocalWorker:
    """
    Local worker for single-node multi-GPU quantization.
    
    Simpler than DistributedGPTQWorker, used when not doing
    multi-node distributed quantization.
    """
    
    def __init__(self, worker_id: int, device: torch.device, config: QuantizationConfig):
        self.worker_id = worker_id
        self.device = device
        self.config = config
        self.memory_manager = GPUMemoryManager(device)
        
        logger.info(f"Initialized local worker {worker_id} on device {device}")
    
    def quantize_layer(
        self,
        layer: nn.Module,
        layer_name: str,
        calibration_data: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Quantize a single layer locally.
        
        Args:
            layer: Layer to quantize
            layer_name: Name of the layer
            calibration_data: Calibration data
            
        Returns:
            Tuple of (quantized_weight, stats)
        """
        start_time = time.time()
        
        try:
            layer = layer.to(self.device)
            
            # Initialize GPTQ
            gptq = GPTQ(
                layer=layer,
                layer_name=layer_name,
                bits=self.config.bits,
                group_size=self.config.group_size,
                actorder=self.config.actorder,
                percdamp=self.config.percdamp,
                device=self.device
            )
            
            # Add calibration data
            for batch in calibration_data:
                batch = batch.to(self.device)
                gptq.add_batch(batch)
            
            # Quantize
            quantized_weight = gptq.quantize()
            
            # Calculate statistics
            stats = {
                'compression_ratio': layer.weight.element_size() * 8 / self.config.bits,
                'quantization_error': torch.mean((layer.weight - quantized_weight) ** 2).item(),
                'processing_time': time.time() - start_time,
                'memory_used_mb': self.memory_manager.get_memory_info()['allocated']
            }
            
            return quantized_weight, stats
            
        finally:
            self.memory_manager.clear_cache()


if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    if len(sys.argv) < 6:
        print("Usage: python worker.py <worker_id> <device> <rank> <world_size> <master_port>")
        sys.exit(1)
    
    worker_id = int(sys.argv[1])
    device = sys.argv[2]
    rank = int(sys.argv[3])
    world_size = int(sys.argv[4])
    master_port = sys.argv[5]
    
    config = QuantizationConfig(bits=4, group_size=128)
    
    start_worker_process(
        worker_id=worker_id,
        device=device,
        config=config,
        world_size=world_size,
        rank=rank,
        master_addr='localhost',
        master_port=master_port
    )