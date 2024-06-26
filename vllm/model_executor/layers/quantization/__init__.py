from typing import Type

from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.marlin import MarlinConfig
from vllm.model_executor.layers.quantization.squeezellm import SqueezeLLMConfig
from vllm.model_executor.layers.quantization.atom import AtomConfig
from vllm.model_executor.layers.quantization.nf4 import NF4Config

_QUANTIZATION_CONFIG_REGISTRY = {
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "squeezellm": SqueezeLLMConfig,
    "marlin": MarlinConfig,
    "atom": AtomConfig,
    "nf4": NF4Config,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
]
