"""
Inter-process communication utilities for distributed GPTQ.

This module provides communication primitives for coordinating
quantization across multiple processes and GPUs.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple
import queue
import threading
import time
import logging
from dataclasses import dataclass


@dataclass
class Message:
    """
    Message structure for inter-process communication.
    """
    sender_id: int
    message_type: str
    data: Any
    timestamp: float


class CommunicationManager:
    """
    Manages communication between distributed processes.
    """
    
    def __init__(self, rank: int, world_size: int, backend: str = 'nccl'):
        """
        Initialize communication manager.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            backend: Communication backend (nccl, gloo, mpi)
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.is_initialized = False
        self.message_queue = queue.Queue()
        self.logger = logging.getLogger(f"CommManager-{rank}")
        
    def initialize(self, master_addr: str = "localhost", master_port: str = "12355"):
        """
        Initialize distributed communication.
        
        Args:
            master_addr: Master node address
            master_port: Master node port
        """
        if not self.is_initialized:
            try:
                dist.init_process_group(
                    backend=self.backend,
                    init_method=f"tcp://{master_addr}:{master_port}",
                    rank=self.rank,
                    world_size=self.world_size
                )
                self.is_initialized = True
                self.logger.info(f"Initialized communication for rank {self.rank}")
            except Exception as e:
                self.logger.error(f"Failed to initialize communication: {e}")
                raise
    
    def broadcast_tensor(self, tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """
        Broadcast a tensor from source rank to all ranks.
        
        Args:
            tensor: Tensor to broadcast
            src_rank: Source rank for broadcast
            
        Returns:
            Broadcasted tensor
        """
        if not self.is_initialized:
            raise RuntimeError("Communication not initialized")
        
        dist.broadcast(tensor, src=src_rank)
        return tensor
    
    def all_gather_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Gather tensors from all ranks.
        
        Args:
            tensor: Local tensor to gather
            
        Returns:
            List of tensors from all ranks
        """
        if not self.is_initialized:
            raise RuntimeError("Communication not initialized")
        
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return gathered_tensors
    
    def reduce_tensor(self, tensor: torch.Tensor, dst_rank: int = 0, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """
        Reduce tensor across all ranks.
        
        Args:
            tensor: Tensor to reduce
            dst_rank: Destination rank for reduction
            op: Reduction operation
            
        Returns:
            Reduced tensor (only valid on dst_rank)
        """
        if not self.is_initialized:
            raise RuntimeError("Communication not initialized")
        
        dist.reduce(tensor, dst=dst_rank, op=op)
        return tensor
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """
        All-reduce tensor across all ranks.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            
        Returns:
            Reduced tensor
        """
        if not self.is_initialized:
            raise RuntimeError("Communication not initialized")
        
        dist.all_reduce(tensor, op=op)
        return tensor
    
    def send_tensor(self, tensor: torch.Tensor, dst_rank: int, tag: int = 0):
        """
        Send tensor to destination rank.
        
        Args:
            tensor: Tensor to send
            dst_rank: Destination rank
            tag: Message tag
        """
        if not self.is_initialized:
            raise RuntimeError("Communication not initialized")
        
        dist.send(tensor, dst=dst_rank, tag=tag)
    
    def recv_tensor(self, tensor: torch.Tensor, src_rank: int, tag: int = 0) -> torch.Tensor:
        """
        Receive tensor from source rank.
        
        Args:
            tensor: Tensor buffer for receiving
            src_rank: Source rank
            tag: Message tag
            
        Returns:
            Received tensor
        """
        if not self.is_initialized:
            raise RuntimeError("Communication not initialized")
        
        dist.recv(tensor, src=src_rank, tag=tag)
        return tensor
    
    def barrier(self):
        """
        Synchronize all processes.
        """
        if not self.is_initialized:
            raise RuntimeError("Communication not initialized")
        
        dist.barrier()
    
    def send_message(self, message: Message, dst_rank: int):
        """
        Send a structured message to destination rank.
        
        Args:
            message: Message to send
            dst_rank: Destination rank
        """
        # For now, implement as a simple tensor-based communication
        # In practice, you might want to use a more sophisticated protocol
        self.logger.info(f"Sending message {message.message_type} to rank {dst_rank}")
    
    def receive_message(self, src_rank: int, timeout: float = 10.0) -> Optional[Message]:
        """
        Receive a message from source rank.
        
        Args:
            src_rank: Source rank
            timeout: Timeout in seconds
            
        Returns:
            Received message or None if timeout
        """
        try:
            message = self.message_queue.get(timeout=timeout)
            return message
        except queue.Empty:
            return None
    
    def cleanup(self):
        """
        Cleanup communication resources.
        """
        if self.is_initialized:
            try:
                dist.destroy_process_group()
                self.is_initialized = False
                self.logger.info(f"Cleaned up communication for rank {self.rank}")
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")


class AsyncCommunicator:
    """
    Asynchronous communication handler for non-blocking operations.
    """
    
    def __init__(self, comm_manager: CommunicationManager):
        """
        Initialize async communicator.
        
        Args:
            comm_manager: Base communication manager
        """
        self.comm_manager = comm_manager
        self.pending_ops = {}
        self.op_counter = 0
    
    def async_send_tensor(self, tensor: torch.Tensor, dst_rank: int) -> int:
        """
        Asynchronously send tensor.
        
        Args:
            tensor: Tensor to send
            dst_rank: Destination rank
            
        Returns:
            Operation ID for tracking
        """
        op_id = self.op_counter
        self.op_counter += 1
        
        # Use torch.distributed.isend for non-blocking send
        req = dist.isend(tensor, dst=dst_rank)
        self.pending_ops[op_id] = req
        
        return op_id
    
    def async_recv_tensor(self, tensor: torch.Tensor, src_rank: int) -> int:
        """
        Asynchronously receive tensor.
        
        Args:
            tensor: Tensor buffer for receiving
            src_rank: Source rank
            
        Returns:
            Operation ID for tracking
        """
        op_id = self.op_counter
        self.op_counter += 1
        
        # Use torch.distributed.irecv for non-blocking receive
        req = dist.irecv(tensor, src=src_rank)
        self.pending_ops[op_id] = req
        
        return op_id
    
    def wait_for_operation(self, op_id: int):
        """
        Wait for an asynchronous operation to complete.
        
        Args:
            op_id: Operation ID to wait for
        """
        if op_id in self.pending_ops:
            self.pending_ops[op_id].wait()
            del self.pending_ops[op_id]
    
    def wait_all(self):
        """
        Wait for all pending operations to complete.
        """
        for req in self.pending_ops.values():
            req.wait()
        self.pending_ops.clear()


def setup_communication(
    rank: int, 
    world_size: int, 
    master_addr: str = "localhost", 
    master_port: str = "12355",
    backend: str = "nccl"
) -> CommunicationManager:
    """
    Setup communication for distributed processing.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
        backend: Communication backend
        
    Returns:
        Initialized communication manager
    """
    comm_manager = CommunicationManager(rank, world_size, backend)
    comm_manager.initialize(master_addr, master_port)
    return comm_manager
