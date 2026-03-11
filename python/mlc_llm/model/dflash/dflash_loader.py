"""
This file specifies how MLC's DFlash parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .dflash_model import DFlashConfig, DFlashDraftModel


def huggingface(model_config: DFlashConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : DFlashConfig
        The configuration of the DFlash draft model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = DFlashDraftModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        # DFlash attention uses separate q/k/v projections — map directly (no fusion)
        attn = f"layers.{i}.self_attn"
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mlc_name = f"{attn}.{proj}.weight"
            if mlc_name in named_parameters:
                mlc_param = named_parameters[mlc_name]
                mapping.add_mapping(
                    mlc_name,
                    [f"{attn}.{proj}.weight"],
                    functools.partial(
                        lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype
                    ),
                )

        # MLP: fuse gate_proj + up_proj into gate_up_proj (like Qwen3)
        mlp = f"layers.{i}.mlp"
        mlc_gate_up_name = f"{mlp}.gate_up_proj.weight"
        if mlc_gate_up_name in named_parameters:
            mlc_param = named_parameters[mlc_gate_up_name]
            mapping.add_mapping(
                mlc_gate_up_name,
                [
                    f"{mlp}.gate_proj.weight",
                    f"{mlp}.up_proj.weight",
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(
                        dtype
                    ),
                    dtype=mlc_param.dtype,
                ),
            )

        # Mark rotary_emb.inv_freq as unused
        mapping.add_unused(f"{attn}.rotary_emb.inv_freq")

    # Mark top-level rotary_emb as unused
    mapping.add_unused("rotary_emb.inv_freq")

    # Map remaining parameters directly (HF names match MLC names for DFlash)
    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype
                ),
            )

    return mapping
