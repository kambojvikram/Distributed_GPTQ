"""
Distributed GPTQ: Fast and efficient quantization for large language models.

Supports:
- Single GPU quantization
- Multi-GPU quantization (data parallel)
- Distributed quantization across multiple nodes
- 2/3/4/8-bit quantization
"""

from .__version__ import __version__
from .core.gptq import GPTQ
from .core.quantizer import DistributedGPTQuantizer, QuantizationConfig, quantize_model_simple
from .distributed.coordinator import (
    DistributedConfig,
    QuantizationMode,
    create_distributed_config,
    DistributedGPTQCoordinator
)

__all__ = [
    "__version__",
    "GPTQ",
    "DistributedGPTQuantizer",
    "QuantizationConfig",
    "quantize_model_simple",
    "DistributedConfig",
    "QuantizationMode",
    "create_distributed_config",
    "DistributedGPTQCoordinator",
]