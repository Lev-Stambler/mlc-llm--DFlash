"""
Implementation for DFlash draft model architecture.
DFlash uses block diffusion for speculative decoding — a lightweight draft model
generates an entire block of tokens in a single parallel forward pass.
"""

import dataclasses
from typing import Any, Dict, List, Optional

from tvm import tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.qwen3.qwen3_model import Qwen3MLP
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DFlashConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the DFlash draft model."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int
    rms_norm_eps: float
    hidden_act: str
    rope_theta: int
    vocab_size: int
    attention_bias: bool = False
    block_size: int = 16
    num_target_layers: int = 36
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    dtype: str = "float32"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # Extract DFlash-specific config from nested dflash_config
        dflash_cfg = self.kwargs.get("dflash_config", {})
        self.mask_token_id: int = dflash_cfg.get("mask_token_id", 151669)
        self.target_layer_ids: List[int] = dflash_cfg.get(
            "target_layer_ids", [1, 9, 17, 25, 33]
        )
        self.num_target_feature_layers = len(self.target_layer_ids)

        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                self.context_window_size = 40960
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = min(self.context_window_size, 2048)


# pylint: disable=invalid-name,missing-docstring


class DFlashAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DFlash cross-attention: Q from draft tokens, K/V from both target context and draft tokens.
    Uses explicit matmul/softmax instead of PagedKVCache."""

    def __init__(self, config: DFlashConfig):
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.num_kv_groups = self.num_q_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_q_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_q_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(config.head_dim, -1, config.rms_norm_eps, bias=False)
        self.k_norm = nn.RMSNorm(config.head_dim, -1, config.rms_norm_eps, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        target_hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        d = self.head_dim
        h_q = self.num_q_heads
        h_kv = self.num_kv_heads
        b, q_len, _ = hidden_states.shape
        _, kv_len, _ = target_hidden.shape

        # Q from draft tokens
        q = self.q_proj(hidden_states)
        q = op.reshape(q, (b, q_len, h_q, d))
        q = self.q_norm(q)

        # K/V from target context
        k_ctx = self.k_proj(target_hidden)
        v_ctx = self.v_proj(target_hidden)
        # K/V from draft tokens
        k_draft = self.k_proj(hidden_states)
        v_draft = self.v_proj(hidden_states)

        # Concatenate context + draft for K/V
        k = op.concat([k_ctx, k_draft], dim=1)  # [b, kv_len+q_len, h_kv*d]
        v = op.concat([v_ctx, v_draft], dim=1)

        total_kv = kv_len + q_len
        k = op.reshape(k, (b, total_kv, h_kv, d))
        v = op.reshape(v, (b, total_kv, h_kv, d))
        k = self.k_norm(k)

        # Apply RoPE - cos/sin shape: [b, total_kv, 1, d]
        # q uses the last q_len positions
        q_cos = op.take(cos, op.arange(kv_len, kv_len + q_len, dtype="int32"), axis=1)
        q_sin = op.take(sin, op.arange(kv_len, kv_len + q_len, dtype="int32"), axis=1)
        q = _apply_rope(q, q_cos, q_sin)
        k = _apply_rope(k, cos, sin)

        # Transpose to [b, heads, seq, d]
        q = op.permute_dims(q, [0, 2, 1, 3])  # [b, h_q, q_len, d]
        k = op.permute_dims(k, [0, 2, 1, 3])  # [b, h_kv, total_kv, d]
        v = op.permute_dims(v, [0, 2, 1, 3])  # [b, h_kv, total_kv, d]

        # GQA: repeat K/V heads
        if self.num_kv_groups > 1:
            k = op.repeat(k, self.num_kv_groups, axis=1)
            v = op.repeat(v, self.num_kv_groups, axis=1)

        # Explicit attention: scores = Q @ K^T / sqrt(d)
        scores = op.matmul(q, op.permute_dims(k, [0, 1, 3, 2]))
        scores = scores * self.scaling
        scores = scores + attention_mask.astype(scores.dtype)
        probs = op.softmax(scores, axis=-1)
        output = op.matmul(probs, v)  # [b, h_q, q_len, d]

        # Reshape back
        output = op.permute_dims(output, [0, 2, 1, 3])  # [b, q_len, h_q, d]
        output = op.reshape(output, (b, q_len, h_q * d))
        return self.o_proj(output)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings. x: [b, s, h, d], cos/sin: [b, s, 1, d]."""
    # Cast cos/sin to match x dtype (they may be float32 from C++ construction)
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = op.concat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig):
        self.self_attn = DFlashAttention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

        def _set_tp():
            def _set(layer, hint):
                layer.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_kv_heads * hd
            v = self.self_attn.num_kv_heads * hd
            i = self.mlp.intermediate_size
            _set(
                self.self_attn.q_proj.weight,
                tp.ShardSingleDim("_shard_q", dim=0),
            )
            _set(
                self.self_attn.k_proj.weight,
                tp.ShardSingleDim("_shard_k", dim=0),
            )
            _set(
                self.self_attn.v_proj.weight,
                tp.ShardSingleDim("_shard_v", dim=0),
            )
            _set(self.self_attn.o_proj.weight, tp.ShardSingleDim("_shard_o", dim=1))
            _set(
                self.mlp.gate_up_proj.weight,
                tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0),
            )
            _set(self.mlp.down_proj.weight, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    def forward(
        self,
        hidden_states: Tensor,
        target_hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, cos, sin, attention_mask)
        hidden_states = self._apply_residual(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self._apply_residual(hidden_states, residual)
        return hidden_states

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class DFlashDraftModel(nn.Module):  # pylint: disable=too-many-instance-attributes
    """DFlash draft model. Stateless — no KV cache.
    Uses target model's embed_tokens and lm_head (shared)."""

    def __init__(self, config: DFlashConfig):
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.fc = nn.Linear(
            config.num_target_feature_layers * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.block_size = config.block_size
        self.rope_theta = config.rope_theta
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def project_target_hidden(self, target_hidden: Tensor) -> Tensor:
        """Project concatenated multi-layer target hidden states to hidden_size."""
        return self.hidden_norm(self.fc(target_hidden))

    def draft_forward(
        self,
        noise_embedding: Tensor,
        projected_target_hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Single-pass block draft forward. Returns hidden states for all block positions."""
        hidden_states = noise_embedding
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, projected_target_hidden, cos, sin, attention_mask
            )
        return self.norm(hidden_states)

    def batch_draft_forward(
        self,
        noise_embedding: Tensor,
        projected_target_hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Batched version of draft_forward."""
        op_ext.configure()
        return self.draft_forward(
            noise_embedding, projected_target_hidden, cos, sin, attention_mask
        )

    def get_default_spec(self):
        mod_spec = {
            "project_target_hidden": {
                "target_hidden": nn.spec.Tensor(
                    [1, "seq_len", self.hidden_size * len([1, 9, 17, 25, 33])],
                    self.dtype,
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "draft_forward": {
                "noise_embedding": nn.spec.Tensor(
                    [1, "block_size", self.hidden_size], self.dtype
                ),
                "projected_target_hidden": nn.spec.Tensor(
                    [1, "ctx_len", self.hidden_size], self.dtype
                ),
                "cos": nn.spec.Tensor([1, "total_len", 1, self.head_dim], "float32"),
                "sin": nn.spec.Tensor([1, "total_len", 1, self.head_dim], "float32"),
                "attention_mask": nn.spec.Tensor(
                    [1, 1, "block_size", "total_len"], "float32"
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_draft_forward": {
                "noise_embedding": nn.spec.Tensor(
                    ["batch_size", "block_size", self.hidden_size], self.dtype
                ),
                "projected_target_hidden": nn.spec.Tensor(
                    ["batch_size", "ctx_len", self.hidden_size], self.dtype
                ),
                "cos": nn.spec.Tensor(
                    ["batch_size", "total_len", 1, self.head_dim], "float32"
                ),
                "sin": nn.spec.Tensor(
                    ["batch_size", "total_len", 1, self.head_dim], "float32"
                ),
                "attention_mask": nn.spec.Tensor(
                    ["batch_size", 1, "block_size", "total_len"], "float32"
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
