# pylint: disable=invalid-name,missing-docstring
"""Unit tests for DFlash draft model architecture."""

import pytest

from mlc_llm.model import MODEL_PRESETS, MODELS
from mlc_llm.model.dflash.dflash_model import DFlashConfig
from mlc_llm.model.qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel
from mlc_llm.serve.config import EngineConfig


# ---------------------------------------------------------------------------
# Inline config dict matching HF structure for z-lab/Qwen3-8B-DFlash-b16
# ---------------------------------------------------------------------------
DFLASH_8B_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 4,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "rope_theta": 1000000,
    "vocab_size": 151936,
    "attention_bias": False,
    "block_size": 16,
    "num_target_layers": 36,
    "max_position_embeddings": 40960,
    "model_type": "dflash",
    "dflash_config": {
        "mask_token_id": 151669,
        "target_layer_ids": [1, 9, 17, 25, 33],
    },
    "context_window_size": 2048,
    "prefill_chunk_size": 2048,
}


def test_dflash_model_registered():
    """Verify DFlash model is in the registry."""
    assert "dflash" in MODELS, "dflash should be registered in MODELS"


def test_dflash_config_from_dict():
    """Create DFlashConfig from a raw dict and verify fields."""
    config = DFlashConfig.from_dict(DFLASH_8B_CONFIG)

    # Standard fields
    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 4
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.vocab_size == 151936
    assert config.intermediate_size == 11008
    assert config.rms_norm_eps == 1e-6
    assert config.hidden_act == "silu"
    assert config.rope_theta == 1000000

    # DFlash-specific fields from nested dflash_config
    assert config.mask_token_id == 151669
    assert config.target_layer_ids == [1, 9, 17, 25, 33]
    assert config.num_target_feature_layers == 5


def test_dflash_config_defaults_without_nested():
    """DFlashConfig should use defaults when dflash_config is absent."""
    config_dict = dict(DFLASH_8B_CONFIG)
    del config_dict["dflash_config"]
    config = DFlashConfig.from_dict(config_dict)

    # Should fall back to defaults
    assert config.mask_token_id == 151669
    assert config.target_layer_ids == [1, 9, 17, 25, 33]
    assert config.num_target_feature_layers == 5


def test_dflash_model_creation():
    """Instantiate DFlashDraftModel and export to TVM IR."""
    model_info = MODELS["dflash"]
    config = model_info.config.from_dict(DFLASH_8B_CONFIG)
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )

    assert mod is not None
    assert len(named_params) > 0

    mod.show(black_format=False)

    for name, param in named_params:
        print(name, param.shape, param.dtype)


def test_dflash_config_validation():
    """Test DFlash configuration has required fields."""
    config = DFlashConfig.from_dict(DFLASH_8B_CONFIG)

    assert hasattr(config, "hidden_size") and config.hidden_size > 0
    assert hasattr(config, "num_hidden_layers") and config.num_hidden_layers > 0
    assert hasattr(config, "num_attention_heads") and config.num_attention_heads > 0
    assert hasattr(config, "vocab_size") and config.vocab_size > 0
    assert hasattr(config, "block_size") and config.block_size > 0

    print(
        f"DFlash Config: hidden_size={config.hidden_size}, "
        f"layers={config.num_hidden_layers}, "
        f"heads={config.num_attention_heads}, "
        f"vocab={config.vocab_size}, "
        f"block_size={config.block_size}"
    )


def test_dflash_speculative_mode_accepted():
    """Verify EngineConfig accepts speculative_mode='dflash' without raising."""
    engine_config = EngineConfig(speculative_mode="dflash")
    assert engine_config.speculative_mode == "dflash"


def test_qwen3_dflash_target_layers_must_be_sorted():
    """DFlash target-layer extraction relies on ascending layer ids."""

    config = Qwen3Config.from_dict(
        {
            "hidden_act": "silu",
            "hidden_size": 128,
            "intermediate_size": 256,
            "attention_bias": False,
            "num_attention_heads": 4,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000,
            "vocab_size": 32000,
            "tie_word_embeddings": True,
            "head_dim": 32,
            "max_position_embeddings": 128,
            "dflash_target_layer_ids": [2, 0],
        }
    )

    with pytest.raises(ValueError, match="ascending order"):
        Qwen3LMHeadModel(config)


if __name__ == "__main__":
    test_dflash_model_registered()
    test_dflash_config_from_dict()
    test_dflash_config_defaults_without_nested()
    test_dflash_model_creation()
    test_dflash_config_validation()
    test_dflash_speculative_mode_accepted()
