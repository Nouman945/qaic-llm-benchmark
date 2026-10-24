# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Model facts the runner needs before it compiles anything: KV cache size per token,
weight size and the longest context, read from the Hub config and the safetensors
metadata (no weights are downloaded).
"""

from dataclasses import asdict, dataclass

DEVICE_MEMORY_GB = 32  # one AI 100 Ultra SoC; a card has four
FIT_FRACTION = 0.9  # headroom for activations and runtime buffers
# MXFP6: 6-bit values plus one 8-bit scale per 32 values = 6.25 bits per weight
BYTES_PER_PARAM = {"fp16": 2.0, "mxfp6": 6.25 / 8}


@dataclass
class ModelInfo:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    params_b: float = None  # None when the safetensors metadata could not be read

    def as_dict(self):
        return asdict(self)


def text_config(config):
    """Vision-language configs keep the decoder settings under text_config."""
    return getattr(config, "text_config", None) or config


def count_params_b(model_name):
    from huggingface_hub import get_safetensors_metadata

    metadata = get_safetensors_metadata(model_name)
    return round(sum(metadata.parameter_count.values()) / 1e9, 2)


def load_model_info(model_name, params_b=None):
    """params_b overrides the safetensors count, which is wrong for packed quantized checkpoints."""
    from transformers import AutoConfig

    config = text_config(AutoConfig.from_pretrained(model_name, trust_remote_code=True))
    heads = config.num_attention_heads
    if params_b is None:
        try:
            params_b = count_params_b(model_name)
        except Exception as exc:  # gated repo, no network, no safetensors: the fit check is skipped
            print(f"warning: {model_name}: no parameter count ({exc}), set params_b in models.json")
    return ModelInfo(
        num_layers=config.num_hidden_layers,
        num_kv_heads=getattr(config, "num_key_value_heads", None) or heads,
        head_dim=getattr(config, "head_dim", None) or config.hidden_size // heads,
        max_position_embeddings=getattr(config, "max_position_embeddings", None),
        params_b=params_b,
    )


def kv_bytes_per_token(info, mxint8_kv_cache):
    """Keys and values of every layer; counts full attention in every layer, so sliding-window models are over-estimated."""
    return 2 * info.num_layers * info.num_kv_heads * info.head_dim * (1 if mxint8_kv_cache else 2)


def estimate_memory_gb(info, case):
    """Weights plus the KV cache of every user at full context, None without a parameter count."""
    if info is None or info.params_b is None:
        return None
    weights = info.params_b * 1e9 * BYTES_PER_PARAM["mxfp6" if case.get("mxfp6") else "fp16"]
    kv = kv_bytes_per_token(info, case.get("mxint8_kv_cache")) * case["ctx_len"] * case.get("batch_size", 1)
    return round((weights + kv) / 1e9, 1)


def memory_budget_gb(num_devices):
    return num_devices * DEVICE_MEMORY_GB * FIT_FRACTION


def fits(estimate_gb, num_devices):
    return estimate_gb is None or estimate_gb <= memory_budget_gb(num_devices)
