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
from tvm.relax.frontend.nn.op import wrap_nested
from tvm.relax.op import strided_slice as _strided_slice

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
    target_model_dtype: str = "float16"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # Extract DFlash-specific config from nested dflash_config
        dflash_cfg = self.kwargs.get("dflash_config", {})
        self.mask_token_id: int = dflash_cfg.get("mask_token_id", 151669)
        self.target_layer_ids: List[int] = dflash_cfg.get(
            "target_layer_ids", [1, 9, 17, 25, 33]
        )
        # Override target_model_dtype from dflash_config if present
        if "target_model_dtype" in dflash_cfg:
            self.target_model_dtype = dflash_cfg["target_model_dtype"]
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

        # Reshape and norm K before RoPE (apply to ctx and draft separately
        # to avoid symbolic shape mismatch with cos/sin in TVM)
        k_ctx = op.reshape(k_ctx, (b, kv_len, h_kv, d))
        k_draft = op.reshape(k_draft, (b, q_len, h_kv, d))
        k_ctx = self.k_norm(k_ctx)
        k_draft = self.k_norm(k_draft)

        # Apply RoPE separately: ctx uses positions 0..kv_len-1,
        # draft uses positions kv_len..kv_len+q_len-1.
        # Use tvm.relax.op.strided_slice with assume_inbound=True so the shape
        # simplifier can reduce ceildiv(kv_len - 0, 1) -> kv_len cleanly.
        ctx_cos = wrap_nested(
            _strided_slice(cos._expr, axes=[1], begin=[0], end=[kv_len],
                           assume_inbound=True),
            name="ctx_cos",
        )
        ctx_sin = wrap_nested(
            _strided_slice(sin._expr, axes=[1], begin=[0], end=[kv_len],
                           assume_inbound=True),
            name="ctx_sin",
        )
        draft_cos = wrap_nested(
            _strided_slice(cos._expr, axes=[1], begin=[kv_len], end=[kv_len + q_len],
                           assume_inbound=True),
            name="draft_cos",
        )
        draft_sin = wrap_nested(
            _strided_slice(sin._expr, axes=[1], begin=[kv_len], end=[kv_len + q_len],
                           assume_inbound=True),
            name="draft_sin",
        )

        q = _apply_rope(q, draft_cos, draft_sin)
        k_ctx = _apply_rope(k_ctx, ctx_cos, ctx_sin)
        k_draft = _apply_rope(k_draft, draft_cos, draft_sin)

        # Reshape V parts
        v_ctx = op.reshape(v_ctx, (b, kv_len, h_kv, d))
        v_draft = op.reshape(v_draft, (b, q_len, h_kv, d))

        # Transpose to [b, heads, seq, d]
        q = op.permute_dims(q, [0, 2, 1, 3])  # [b, h_q, q_len, d]
        k_ctx = op.permute_dims(k_ctx, [0, 2, 1, 3])  # [b, h_kv, kv_len, d]
        k_draft = op.permute_dims(k_draft, [0, 2, 1, 3])  # [b, h_kv, q_len, d]
        v_ctx = op.permute_dims(v_ctx, [0, 2, 1, 3])
        v_draft = op.permute_dims(v_draft, [0, 2, 1, 3])

        # GQA: repeat K/V heads
        if self.num_kv_groups > 1:
            k_ctx = op.repeat(k_ctx, self.num_kv_groups, axis=1)
            k_draft = op.repeat(k_draft, self.num_kv_groups, axis=1)
            v_ctx = op.repeat(v_ctx, self.num_kv_groups, axis=1)
            v_draft = op.repeat(v_draft, self.num_kv_groups, axis=1)

        # Concatenate K and V for full attention
        k = op.concat([k_ctx, k_draft], dim=2)  # [b, h_kv/h_q, kv_len+q_len, d]
        v = op.concat([v_ctx, v_draft], dim=2)

        # Explicit attention: scores = Q @ K^T / sqrt(d)
        scores = op.matmul(q, op.permute_dims(k, [0, 1, 3, 2]))  # [b, h_q, q_len, kv_len+q_len]
        scores = scores * self.scaling

        # Reshape scores so last dim matches attention_mask's "total_len" symbol
        total_len = attention_mask.shape[3]
        scores = op.reshape(scores, (b, self.num_q_heads, q_len, total_len))

        scores = scores + attention_mask.astype(scores.dtype)
        probs = op.softmax(scores, axis=-1)

        # Reshape V to match scores' last dim for matmul
        v = op.reshape(v, (b, self.num_q_heads, total_len, d))
        output = op.matmul(probs, v)  # [b, h_q, q_len, d]

        # Reshape back
        output = op.permute_dims(output, [0, 2, 1, 3])  # [b, q_len, h_q, d]
        output = op.reshape(output, (b, q_len, h_q * d))
        return self.o_proj(output)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings. x: [b, s, h, d], cos/sin: [b, s, 1, d]."""
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    x1, x2 = op.split(x, 2, axis=-1)
    rotated = op.concat([op.negative(x2), x1], dim=-1)
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
        self.target_model_dtype = config.target_model_dtype
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def project_target_hidden(self, target_hidden: Tensor) -> Tensor:
        """Project concatenated multi-layer target hidden states to hidden_size.

        The fc matmul reduces 20480 dimensions (5 layers × 4096 hidden) to 4096.
        Target hidden states can have large values (max_abs ~60) from concatenated
        layer activations. In float16, the dot product of 20480 large values
        overflows (sum can exceed 65504). We compute in float32 and keep float32
        through RMSNorm — the matmul output can also exceed 65504 so we must
        normalize before casting back to float16.
        """
        w = self.fc.weight.astype("float32")
        h = target_hidden.astype("float32")
        projected = op.matmul(h, op.permute_dims(w, [1, 0]))
        # RMSNorm in float32: weight is fp16 but op.rms_norm handles the cast
        gamma = self.hidden_norm.weight.astype("float32")
        normalized = op.rms_norm(projected, weight=gamma, axes=[-1], epsilon=self.hidden_norm.epsilon)
        return normalized.astype(self.dtype)

    def draft_forward(
        self,
        noise_embedding: Tensor,
        projected_target_hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Single-pass block draft forward. Returns hidden states for all block positions."""
        # Cast embedding to model dtype (bf16) since it comes from target model (fp16).
        # Internal computation in bf16 for accuracy, output cast to target_model_dtype
        # for target model's get_logits.
        hidden_states = noise_embedding.astype(self.dtype)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, projected_target_hidden, cos, sin, attention_mask
            )
        return self.norm(hidden_states).astype(self.target_model_dtype)

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
                # Input dtype must match target model's activation dtype (not
                # draft model's dtype). The function casts to float32 internally.
                "target_hidden": nn.spec.Tensor(
                    [1, "seq_len", self.hidden_size * len([1, 9, 17, 25, 33])],
                    self.target_model_dtype,
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "draft_forward": {
                # noise_embedding comes from target model's embedder, so its
                # dtype matches the target model, not the draft model.
                "noise_embedding": nn.spec.Tensor(
                    [1, "block_size", self.hidden_size], self.target_model_dtype
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
                    ["batch_size", "block_size", self.hidden_size], self.target_model_dtype
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
