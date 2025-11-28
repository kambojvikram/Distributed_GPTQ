"""
Main interface for distributed GPTQ quantization.
"""

import torch
import torch.nn as nn
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Union, Optional, Callable

from tqdm import tqdm
from safetensors import safe_open, save_file, load_file

from .gptq import GPTQQuantizer
from ..utils.logging_utils import get_logger
from ..distributed import DistributedConfig, create_distributed_config, DistributedGPTQCoordinator
from ..utils import clear_gpu_cache

logger = get_logger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for GPTQ quantization."""
    bits: int = 4
    group_size: int = 128
    actorder: bool = False
    percdamp: float = 0.01
    blocksize: int = 128
    calibration_samples: int = 128
    use_triton: bool = False
    save_packed: bool = True
    
    def validate(self):
        """Validate configuration."""
        assert self.bits in [2, 3, 4, 8], f"Bits must be 2, 3, 4, or 8, got {self.bits}"
        assert self.group_size > 0, f"Group size must be positive, got {self.group_size}"
        assert 0 <= self.percdamp <= 1, f"Percdamp must be between 0 and 1, got {self.percdamp}"


class DistributedGPTQuantizer:
    """
    Main interface for distributed GPTQ quantization.
    
    Example:
        # Single GPU
        quantizer = DistributedGPTQuantizer()
        quantized_model = quantizer.quantize_model(model, calibration_data)
        
        # Multi-GPU (launched with torchrun)
        quantizer = DistributedGPTQuantizer(distributed_config=create_distributed_config())
        quantized_model = quantizer.quantize_model(model, calibration_data)
    """
    
    def __init__(
        self,
        quantization_config: Optional[QuantizationConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        """
        Initialize the quantizer.
        
        Args:
            quantization_config: Quantization configuration
            distributed_config: Distributed training configuration
        """
        self.quant_config = quantization_config or QuantizationConfig()
        self.quant_config.validate()
        
        self.dist_config = distributed_config or create_distributed_config()
        self.coordinator = DistributedGPTQCoordinator(self.dist_config)
        
        # Track quantization progress
        self.quantized_layers = {}
        self.quantization_errors = {}
        
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Union[List[torch.Tensor], torch.utils.data.DataLoader],
        layers_to_quantize: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> nn.Module:
        """
        Quantize a model using distributed GPTQ.
        
        Args:
            model: Model to quantize
            calibration_data: Calibration data for computing Hessians
            layers_to_quantize: Specific layers to quantize (None = all linear layers)
            save_path: Path to save quantized model
            progress_callback: Optional callback for progress updates
            
        Returns:
            Quantized model
        """
        start_time = time.time()
        
        with self.coordinator.distributed_context():
            # Prepare model for distributed quantization
            model = self.coordinator.prepare_model(model)
            model.eval()
            
            # Get layers to quantize
            if layers_to_quantize is None:
                layers_to_quantize = self._find_layers_to_quantize(model)
            
            logger.info(f"Found {len(layers_to_quantize)} layers to quantize")
            
            # Prepare calibration data
            if isinstance(calibration_data, list):
                calibration_loader = self._create_calibration_loader(calibration_data)
            else:
                calibration_loader = calibration_data
                
            # Distribute calibration data across GPUs
            calibration_loader = self.coordinator.distribute_calibration_data(
                calibration_loader
            )
            
            # Collect layer inputs
            layer_inputs = self._collect_layer_inputs(
                model,
                calibration_loader,
                layers_to_quantize
            )
            
            # Quantize each layer
            total_layers = len(layers_to_quantize)
            for idx, (layer_name, layer) in enumerate(layers_to_quantize.items()):
                if self.coordinator.config.rank == 0:
                    logger.info(f"Quantizing layer {idx+1}/{total_layers}: {layer_name}")
                
                # Get inputs for this layer
                inputs = layer_inputs.get(layer_name, [])
                
                # Quantize layer
                stats = self.coordinator.quantize_layer_distributed(
                    layer,
                    inputs,
                    asdict(self.quant_config)
                )
                
                self.quantized_layers[layer_name] = stats
                self.quantization_errors[layer_name] = stats.get('total_error', 0)
                
                # Clear GPU cache
                clear_gpu_cache()
                
                # Progress callback
                if progress_callback and self.coordinator.config.rank == 0:
                    progress_callback(idx + 1, total_layers, layer_name, stats)
                    
            # Log summary
            if self.coordinator.config.rank == 0:
                self._log_quantization_summary(time.time() - start_time)
                
            # Save model if requested
            if save_path:
                self.save_quantized_model(model, save_path)
                
        return model
        
    def _find_layers_to_quantize(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Find all linear layers to quantize."""
        layers = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Skip small layers
                if hasattr(module, 'weight'):
                    weight_numel = module.weight.numel()
                    if weight_numel > 1000:  # Arbitrary threshold
                        layers[name] = module
                        
        return layers
        
    def _create_calibration_loader(
        self,
        calibration_data: List[torch.Tensor]
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader from list of tensors."""
        dataset = torch.utils.data.TensorDataset(*calibration_data)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
    @torch.no_grad()
    def _collect_layer_inputs(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        layers_to_quantize: Dict[str, nn.Module]
    ) -> Dict[str, List[torch.Tensor]]:
        """Collect inputs for each layer during forward pass."""
        layer_inputs = {name: [] for name in layers_to_quantize}
        handles = []
        
        def create_hook(layer_name):
            def hook(module, input, output):
                # Store input
                if isinstance(input, tuple):
                    input = input[0]
                layer_inputs[layer_name].append(input.detach().cpu())
            return hook
            
        # Register hooks
        for name, layer in layers_to_quantize.items():
            handle = layer.register_forward_hook(create_hook(name))
            handles.append(handle)
            
        # Run forward passes
        num_samples = 0
        for batch in tqdm(
            calibration_loader,
            desc="Collecting calibration data",
            disable=self.coordinator.config.rank != 0
        ):
            if num_samples >= self.quant_config.calibration_samples:
                break
                
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.coordinator.device)
            
            # Forward pass
            try:
                _ = model(batch)
            except Exception as e:
                logger.warning(f"Error during forward pass: {e}")
                continue
                
            num_samples += batch.shape[0]
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return layer_inputs
        
    def save_quantized_model(
        self,
        model: nn.Module,
        save_path: str,
        save_format: str = "safetensors"
    ) -> None:
        """
        Save quantized model.
        
        Args:
            model: Quantized model
            save_path: Path to save
            save_format: Format to save in ("safetensors" or "pytorch")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            'quantization_config': asdict(self.quant_config),
            'quantized_layers': list(self.quantized_layers.keys()),
            'quantization_errors': self.quantization_errors,
        }
        
        if save_format == "safetensors":
            # Convert model state dict for safetensors
            state_dict = model.state_dict()
            
            # Save metadata separately
            metadata_path = save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            # Save model
            if self.coordinator.config.rank == 0:
                save_file(state_dict, save_path)
                logger.info(f"Saved quantized model to {save_path}")
        else:
            # PyTorch format
            save_data['model_state_dict'] = model.state_dict()
            
            if self.coordinator.config.rank == 0:
                torch.save(save_data, save_path)
                logger.info(f"Saved quantized model to {save_path}")
                
    def load_quantized_model(
        self,
        model: nn.Module,
        load_path: str,
        load_format: str = "safetensors"
    ) -> nn.Module:
        """
        Load a quantized model.
        
        Args:
            model: Model architecture (unquantized)
            load_path: Path to load from
            load_format: Format to load from
            
        Returns:
            Loaded quantized model
        """
        load_path = Path(load_path)
        
        if load_format == "safetensors":
            # Load metadata
            metadata_path = load_path.with_suffix('.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Load model weights
            state_dict = load_file(load_path)
            model.load_state_dict(state_dict)
            
            # Update config
            self.quant_config = QuantizationConfig(**metadata['quantization_config'])
            self.quantized_layers = {name: {} for name in metadata['quantized_layers']}
            self.quantization_errors = metadata['quantization_errors']
        else:
            # PyTorch format
            checkpoint = torch.load(load_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.quant_config = QuantizationConfig(**checkpoint['quantization_config'])
            self.quantized_layers = {name: {} for name in checkpoint['quantized_layers']}
            self.quantization_errors = checkpoint['quantization_errors']
            
        return model
        
    def _log_quantization_summary(self, elapsed_time: float) -> None:
        """Log quantization summary."""
        total_error = sum(self.quantization_errors.values())
        avg_error = total_error / len(self.quantization_errors) if self.quantization_errors else 0
        
        logger.info("=" * 50)
        logger.info("Quantization Summary:")
        logger.info(f"Total layers quantized: {len(self.quantized_layers)}")
        logger.info(f"Quantization bits: {self.quant_config.bits}")
        logger.info(f"Group size: {self.quant_config.group_size}")
        logger.info(f"Average quantization error: {avg_error:.6f}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Compression ratio: {32.0 / self.quant_config.bits:.2f}x")
        
        # Memory stats
        if self.coordinator.device.type == 'cuda':
            memory_stats = self.coordinator.get_memory_stats()
            logger.info(f"Peak GPU memory: {memory_stats['max_allocated_gb']:.2f} GB")
        logger.info("=" * 50)


def quantize_model_simple(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    bits: int = 4,
    save_path: Optional[str] = None
) -> nn.Module:
    """
    Simple interface for model quantization.
    
    Args:
        model: Model to quantize
        calibration_data: Calibration data
        bits: Quantization bits
        save_path: Optional save path
        
    Returns:
        Quantized model
    """
    config = QuantizationConfig(bits=bits)
    quantizer = DistributedGPTQuantizer(config)
    
    return quantizer.quantize_model(
        model,
        calibration_data,
        save_path=save_path
    )