"""
HuggingFace Transformers model support for distributed GPTQ.

This module provides specialized support for quantizing HuggingFace
Transformers models using distributed GPTQ.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    PreTrainedModel, PreTrainedTokenizer
)

from .base_model import QuantizableModel, ModelInfo


class TransformersModelWrapper(QuantizableModel):
    """
    Wrapper for HuggingFace Transformers models.
    """
    
    def __init__(
        self, 
        model: Union[str, PreTrainedModel], 
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Transformers model wrapper.
        
        Args:
            model: Model name/path or PreTrainedModel instance
            tokenizer: Tokenizer for the model
            config: Configuration dictionary
        """
        if isinstance(model, str):
            self.model = AutoModel.from_pretrained(model)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
        else:
            self.model = model
            self.tokenizer = tokenizer
            
        self.config = config or {}
        self.model_name = getattr(self.model, 'name_or_path', 'unknown')
        self._layer_cache = {}
        self._dependency_cache = None
        
        # Model-specific configurations
        self.model_type = self.model.config.model_type
        self._setup_model_specific_config()
    
    def _setup_model_specific_config(self):
        """Setup model-specific configurations."""
        if self.model_type in ['llama', 'mistral', 'mixtral']:
            self.attention_layers = self._get_attention_layers()
            self.mlp_layers = self._get_mlp_layers()
        elif self.model_type in ['gpt2', 'gpt_neo', 'gpt_neox']:
            self.attention_layers = self._get_gpt_attention_layers()
            self.mlp_layers = self._get_gpt_mlp_layers()
        elif self.model_type == 'bert':
            self.attention_layers = self._get_bert_attention_layers()
            self.mlp_layers = self._get_bert_mlp_layers()
    
    def _get_attention_layers(self) -> List[str]:
        """Get attention layer names for LLaMA-style models."""
        layers = []
        for name, module in self.model.named_modules():
            if 'self_attn' in name and isinstance(module, nn.Linear):
                layers.append(name)
        return layers
    
    def _get_mlp_layers(self) -> List[str]:
        """Get MLP layer names for LLaMA-style models."""
        layers = []
        for name, module in self.model.named_modules():
            if 'mlp' in name and isinstance(module, nn.Linear):
                layers.append(name)
        return layers
    
    def _get_gpt_attention_layers(self) -> List[str]:
        """Get attention layer names for GPT-style models."""
        layers = []
        for name, module in self.model.named_modules():
            if 'attn' in name and isinstance(module, nn.Linear):
                layers.append(name)
        return layers
    
    def _get_gpt_mlp_layers(self) -> List[str]:
        """Get MLP layer names for GPT-style models."""
        layers = []
        for name, module in self.model.named_modules():
            if 'mlp' in name and isinstance(module, nn.Linear):
                layers.append(name)
        return layers
    
    def _get_bert_attention_layers(self) -> List[str]:
        """Get attention layer names for BERT-style models."""
        layers = []
        for name, module in self.model.named_modules():
            if 'attention' in name and isinstance(module, nn.Linear):
                layers.append(name)
        return layers
    
    def _get_bert_mlp_layers(self) -> List[str]:
        """Get MLP layer names for BERT-style models."""
        layers = []
        for name, module in self.model.named_modules():
            if ('intermediate' in name or 'output' in name) and isinstance(module, nn.Linear):
                layers.append(name)
        return layers
    
    def get_layers(self) -> Dict[str, nn.Module]:
        """
        Get all quantizable layers.
        
        Returns:
            Dictionary of layer names to modules
        """
        if not self._layer_cache:
            self._layer_cache = {}
            
            # Get all Linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    # Skip certain layers based on configuration
                    skip_patterns = self.config.get('skip_layers', [])
                    if not any(pattern in name for pattern in skip_patterns):
                        self._layer_cache[name] = module
        
        return self._layer_cache
    
    def get_layer_dependencies(self) -> Dict[str, List[str]]:
        """
        Get layer dependencies based on model architecture.
        
        Returns:
            Dictionary of dependencies
        """
        if self._dependency_cache is None:
            self._dependency_cache = self._analyze_dependencies()
        
        return self._dependency_cache
    
    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze model dependencies."""
        dependencies = {}
        layers = list(self.get_layers().keys())
        
        # Group layers by transformer block
        blocks = {}
        for layer_name in layers:
            # Extract block number/identifier
            parts = layer_name.split('.')
            if 'layers' in parts or 'layer' in parts:
                try:
                    block_idx = None
                    for i, part in enumerate(parts):
                        if part in ['layers', 'layer'] and i + 1 < len(parts):
                            block_idx = int(parts[i + 1])
                            break
                    
                    if block_idx is not None:
                        if block_idx not in blocks:
                            blocks[block_idx] = []
                        blocks[block_idx].append(layer_name)
                except (ValueError, IndexError):
                    pass
        
        # Set dependencies within blocks
        for block_idx, block_layers in blocks.items():
            for i, layer_name in enumerate(block_layers):
                if i == 0 and block_idx > 0:
                    # First layer in block depends on previous block
                    prev_block = blocks.get(block_idx - 1, [])
                    dependencies[layer_name] = prev_block[-1:] if prev_block else []
                elif i > 0:
                    # Layer depends on previous layer in same block
                    dependencies[layer_name] = [block_layers[i - 1]]
                else:
                    dependencies[layer_name] = []
        
        # Handle layers not in blocks
        for layer_name in layers:
            if layer_name not in dependencies:
                dependencies[layer_name] = []
        
        return dependencies
    
    def prepare_calibration_data(self, calibration_dataset) -> torch.utils.data.DataLoader:
        """
        Prepare calibration data for transformers.
        
        Args:
            calibration_dataset: Text dataset or tokenized data
            
        Returns:
            DataLoader with tokenized data
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for calibration data preparation")
        
        batch_size = self.config.get('calibration_batch_size', 1)
        max_length = self.config.get('max_length', 512)
        
        # Tokenize if needed
        if isinstance(calibration_dataset[0], str):
            tokenized_data = []
            for text in calibration_dataset:
                tokens = self.tokenizer(
                    text,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                tokenized_data.append(tokens)
            calibration_dataset = tokenized_data
        
        return torch.utils.data.DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Collate function for tokenized data."""
        if len(batch) == 1:
            return batch[0]
        
        # Batch tokenized inputs
        input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
        attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Add other keys if present
        for key in batch[0].keys():
            if key not in result:
                result[key] = torch.cat([item[key] for item in batch], dim=0)
        
        return result
    
    def forward_to_layer(self, inputs: Dict[str, torch.Tensor], layer_name: str) -> torch.Tensor:
        """
        Forward pass up to a specific layer.
        
        Args:
            inputs: Tokenized inputs
            layer_name: Target layer name
            
        Returns:
            Hidden states at the target layer
        """
        # This requires model tracing or hooks
        # For now, return a simplified implementation
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Try to extract the appropriate hidden state
            # This is model-specific and may need refinement
            if hasattr(outputs, 'hidden_states'):
                return outputs.hidden_states[-1]  # Last hidden state
            else:
                return outputs.last_hidden_state
    
    def replace_layer(self, layer_name: str, new_layer: nn.Module):
        """
        Replace a layer with a quantized version.
        
        Args:
            layer_name: Name of layer to replace
            new_layer: New quantized layer
        """
        # Navigate to the parent module and replace the layer
        parts = layer_name.split('.')
        parent = self.model
        
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        if parts[-1].isdigit():
            parent[int(parts[-1])] = new_layer
        else:
            setattr(parent, parts[-1], new_layer)
        
        # Update cache
        if layer_name in self._layer_cache:
            self._layer_cache[layer_name] = new_layer
    
    def get_model_info(self) -> ModelInfo:
        """
        Get model information.
        
        Returns:
            ModelInfo object
        """
        return ModelInfo(self.model)
    
    def save_pretrained(self, save_directory: str):
        """Save the model using HuggingFace format."""
        self.model.save_pretrained(save_directory)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_directory)
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode."""
        self.model.train(mode)
        return self
    
    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        self.model = self.model.to(device)
        return self
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)


def load_transformers_model(
    model_name_or_path: str,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = None,
    **kwargs
) -> TransformersModelWrapper:
    """
    Load a HuggingFace Transformers model for quantization.
    
    Args:
        model_name_or_path: Model name or path
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        **kwargs: Additional arguments for model loading
        
    Returns:
        Wrapped Transformers model
    """
    # Load model with specified parameters
    model_kwargs = {
        'torch_dtype': torch_dtype or torch.float16,
        'device_map': device_map,
        **kwargs
    }
    
    model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    return TransformersModelWrapper(model, tokenizer)
