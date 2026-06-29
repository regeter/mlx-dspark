"""DSpark drafter config — loaded from the HF checkpoint's config.json.

Supports two drafter families with a shared inference path:
  - gemma4  (gemma4_text): k_eq_v attention, v_norm, partial/proportional rope,
            sandwich norms + layer_scalar, gelu-tanh MLP, logit softcap.
  - qwen3   (qwen3):       standard GQA (separate v_proj, no v_norm), default rope,
            Llama-style 2-norm layer, silu MLP, no softcap.
Only the fields the MLX inference path needs are pulled out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DSparkConfig:
    family: str = "gemma4"             # "gemma4" | "qwen3"

    # core dims
    hidden_size: int = 3840
    vocab_size: int = 262144
    num_hidden_layers: int = 5
    intermediate_size: int = 15360
    rms_norm_eps: float = 1e-6

    # attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    attention_k_eq_v: bool = True
    attention_bias: bool = False

    # rope
    rope_theta: float = 1_000_000.0
    partial_rotary_factor: float = 0.25
    rope_type: str = "proportional"

    # dspark specifics
    block_size: int = 7
    mask_token_id: int = 4
    target_layer_ids: list[int] = field(default_factory=lambda: [5, 17, 29, 41, 46])
    num_target_layers: int = 48

    # markov + confidence
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    # logits
    final_logit_softcapping: float | None = 30.0
    pad_token_id: int = 0

    # ---- family-derived knobs (set in from_json) ----
    mlp_activation: str = "gelu_tanh"   # "gelu_tanh" | "silu"
    norm_style: str = "gemma"           # "gemma" (sandwich+scalar) | "qwen" (llama 2-norm)
    use_v_norm: bool = True             # gemma: RMSNormNoScale v_norm; qwen: none
    attention_scaling: float | None = None  # None -> 1/sqrt(attn_head_dim)

    @property
    def attn_head_dim(self) -> int:
        """Head dim used by the drafter's own attention."""
        return self.global_head_dim if self.family == "gemma4" else self.head_dim

    @property
    def n_kv_heads(self) -> int:
        if self.family == "gemma4" and self.attention_k_eq_v:
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    @property
    def scaling(self) -> float:
        if self.attention_scaling is not None:
            return self.attention_scaling
        return self.attn_head_dim ** -0.5 if self.family == "qwen3" else 1.0

    @property
    def rope_parameters(self) -> dict:
        return {"rope_type": self.rope_type, "partial_rotary_factor": self.partial_rotary_factor}

    @classmethod
    def from_json(cls, path: str | Path) -> "DSparkConfig":
        with open(path) as f:
            c = json.load(f)
        mt = c.get("model_type", "")
        family = "qwen3" if "qwen3" in mt else "gemma4"

        if family == "qwen3":
            rp = c.get("rope_parameters") or {}
            return cls(
                family="qwen3",
                hidden_size=c["hidden_size"], vocab_size=c["vocab_size"],
                num_hidden_layers=c["num_hidden_layers"],
                intermediate_size=c["intermediate_size"],
                rms_norm_eps=c.get("rms_norm_eps", 1e-6),
                num_attention_heads=c["num_attention_heads"],
                num_key_value_heads=c.get("num_key_value_heads", 8),
                head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
                attention_k_eq_v=False, attention_bias=c.get("attention_bias", False),
                rope_theta=rp.get("rope_theta", c.get("rope_theta", 1_000_000.0)),
                rope_type="default",
                block_size=c["block_size"], mask_token_id=c["mask_token_id"],
                target_layer_ids=list(c["target_layer_ids"]),
                num_target_layers=c.get("num_target_layers", 36),
                markov_rank=c.get("markov_rank", 256),
                markov_head_type=c.get("markov_head_type", "vanilla"),
                enable_confidence_head=c.get("enable_confidence_head", True),
                confidence_head_with_markov=c.get("confidence_head_with_markov", True),
                final_logit_softcapping=c.get("final_logit_softcapping", None),
                pad_token_id=c.get("pad_token_id") or 0,
                mlp_activation="silu", norm_style="qwen", use_v_norm=False,
            )

        rope = (c.get("rope_parameters") or {}).get("full_attention", {}) or {}
        return cls(
            family="gemma4",
            hidden_size=c["hidden_size"], vocab_size=c["vocab_size"],
            num_hidden_layers=c["num_hidden_layers"],
            intermediate_size=c["intermediate_size"],
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", 8),
            num_global_key_value_heads=c.get("num_global_key_value_heads", 1),
            head_dim=c.get("head_dim", 256), global_head_dim=c.get("global_head_dim", 512),
            attention_k_eq_v=c.get("attention_k_eq_v", True),
            attention_bias=c.get("attention_bias", False),
            rope_theta=rope.get("rope_theta", 1_000_000.0),
            partial_rotary_factor=rope.get("partial_rotary_factor", 0.25),
            rope_type=rope.get("rope_type", "proportional"),
            block_size=c["block_size"], mask_token_id=c["mask_token_id"],
            target_layer_ids=list(c["target_layer_ids"]),
            num_target_layers=c.get("num_target_layers", 48),
            markov_rank=c.get("markov_rank", 256),
            markov_head_type=c.get("markov_head_type", "vanilla"),
            enable_confidence_head=c.get("enable_confidence_head", True),
            confidence_head_with_markov=c.get("confidence_head_with_markov", True),
            final_logit_softcapping=c.get("final_logit_softcapping", 30.0),
            pad_token_id=c.get("pad_token_id", 0),
            mlp_activation="gelu_tanh", norm_style="gemma", use_v_norm=True,
        )
