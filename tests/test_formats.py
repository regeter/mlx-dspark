"""Checkpoint-format robustness: the loader must either load a checkpoint correctly or
refuse it with an error that names the real problem — never silently mis-parse.

Fixtures are condensed from real configs observed on HF (2026-07-07):
- vLLM "speculators" format (RedHatAI/GLM-5.2-speculator.dspark — note model_type "qwen3"!)
- full target model with embedded drafter (deepseek-ai/DeepSeek-V4-Pro-DSpark)
- DFlash+Markov community hybrid (Hikari07jp/DSpark-Gemma-4-31B-draft)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_dspark.config import DSparkConfig
from mlx_dspark.load import _route_target, load_dflash, load_drafter
from mlx_dspark.model import DSparkDrafter
from mlx_dspark.target import Target

# ---------------------------------------------------------------- config fixtures

QWEN3_MIN = {
    "model_type": "qwen3", "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1,
    "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 8, "block_size": 4, "mask_token_id": 3, "target_layer_ids": [0, 1],
    "num_target_layers": 2, "markov_rank": 4,
}

GEMMA4_MIN = {
    "model_type": "gemma4_text", "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 1,
    "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
    "num_global_key_value_heads": 1, "head_dim": 8, "global_head_dim": 8, "block_size": 4,
    "mask_token_id": 3, "target_layer_ids": [0, 1], "num_target_layers": 2, "markov_rank": 4,
}

SPECULATORS = {  # vLLM speculators packaging — drafter backbone is ALSO model_type "qwen3"
    "architectures": ["DSparkDraftModel"], "model_type": "qwen3",
    "speculators_model_type": "dspark",
    "speculators_config": {"algorithm": "dspark",
                           "verifier": {"architectures": ["GlmMoeDsaForCausalLM"]}},
    "block_size": 8, "markov_rank": 256, "aux_hidden_state_layer_ids": [1, 2, 3],
    "transformer_layer_config": {"hidden_size": 6144},
}

EMBEDDED_V4 = {  # full 893 GB target with dspark_* fields — not a standalone drafter
    "model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"],
    "hidden_size": 7168, "num_hidden_layers": 61, "vocab_size": 129280,
    "dspark_block_size": 5, "dspark_target_layer_ids": [58, 59, 60],
    "dspark_markov_rank": 512,
}

DFLASH_MARKOV_HYBRID = {  # DFlash block-16 backbone + a DSpark Markov head
    "model_type": "qwen3", "architectures": ["DFlashDraftModel"],
    "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
    "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 32, "vocab_size": 32,
    "rms_norm_eps": 1e-6, "rope_theta": 1e6, "max_position_embeddings": 1024,
    "block_size": 16, "target_layer_ids": [0, 1], "num_target_layers": 2,
    "markov_rank": 256, "markov_head_type": "vanilla",
}


def _cfg_path(tmp_path, cfg: dict) -> str:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_qwen3_config_parses(tmp_path):
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, QWEN3_MIN))
    assert cfg.family == "qwen3" and cfg.block_size == 4


def test_gemma4_config_parses(tmp_path):
    cfg = DSparkConfig.from_json(_cfg_path(tmp_path, GEMMA4_MIN))
    assert cfg.family == "gemma4" and cfg.target_layer_ids == [0, 1]


def test_speculators_format_refused_with_reason(tmp_path):
    # must be caught BEFORE family detection — model_type says "qwen3"
    with pytest.raises(ValueError, match="speculators"):
        DSparkConfig.from_json(_cfg_path(tmp_path, SPECULATORS))


def test_embedded_full_model_refused_with_reason(tmp_path):
    with pytest.raises(ValueError, match="embedded"):
        DSparkConfig.from_json(_cfg_path(tmp_path, EMBEDDED_V4))


def test_unknown_family_refused_not_silently_gemma(tmp_path):
    with pytest.raises(ValueError, match="unsupported drafter family"):
        DSparkConfig.from_json(_cfg_path(tmp_path, {**QWEN3_MIN, "model_type": "llama"}))


def test_missing_model_type_refused(tmp_path):
    cfg = {k: v for k, v in QWEN3_MIN.items() if k != "model_type"}
    with pytest.raises(ValueError, match="unsupported drafter family"):
        DSparkConfig.from_json(_cfg_path(tmp_path, cfg))


def test_missing_required_field_refused(tmp_path):
    cfg = {k: v for k, v in QWEN3_MIN.items() if k != "target_layer_ids"}
    with pytest.raises(ValueError, match="target_layer_ids"):
        DSparkConfig.from_json(_cfg_path(tmp_path, cfg))


# ---------------------------------------------------------------- target routing

def test_route_multimodal_markers_to_vlm():
    assert _route_target({"model_type": "gemma4_unified", "vision_config": {}}) == "mlx_vlm"
    assert _route_target({"model_type": "whatever", "audio_config": {}}) == "mlx_vlm"


def test_route_mlxlm_families_to_mlxlm():
    for mt in ("qwen3", "qwen3_moe", "llama", "glm_moe_dsa", "deepseek_v3"):
        assert _route_target({"model_type": mt}) == "mlx_lm", mt


def test_route_respects_mlxlm_remap_table():
    assert _route_target({"model_type": "mistral"}) == "mlx_lm"  # remapped to llama


def test_route_unknown_falls_back_to_vlm():
    assert _route_target({"model_type": "totally_unknown_family"}) == "mlx_vlm"
    assert _route_target({}) == "mlx_vlm"


# ---------------------------------------------------------------- strict weight loading

def _write_drafter_ckpt(tmp_path, weights: dict) -> str:
    (tmp_path / "config.json").write_text(json.dumps(QWEN3_MIN))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return str(tmp_path)


def _reference_weights() -> dict:
    class _P:  # only the fields DSparkConfig needs, via a real parse for fidelity
        pass
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfgp = f"{d}/config.json"
        open(cfgp, "w").write(json.dumps(QWEN3_MIN))
        cfg = DSparkConfig.from_json(cfgp)
    ref = DSparkDrafter(cfg)
    weights = dict(tree_flatten(ref.parameters()))
    mx.eval(list(weights.values()))
    return weights


def test_load_drafter_matching_checkpoint_loads(tmp_path):
    path = _write_drafter_ckpt(tmp_path, _reference_weights())
    drafter, cfg = load_drafter(path, quantize=False)
    assert cfg.family == "qwen3" and drafter.block_size == 4


def test_load_drafter_mismatch_raises_by_default(tmp_path):
    weights = _reference_weights()
    weights["bogus.weight"] = weights.pop("lm_head.weight")
    path = _write_drafter_ckpt(tmp_path, weights)
    with pytest.raises(ValueError, match="don't match"):
        load_drafter(path, quantize=False)


def test_load_drafter_strict_false_force_loads(tmp_path):
    weights = _reference_weights()
    weights["bogus.weight"] = weights.pop("lm_head.weight")
    path = _write_drafter_ckpt(tmp_path, weights)
    drafter, _ = load_drafter(path, quantize=False, strict=False)  # warn-and-load
    assert drafter is not None


def test_load_dflash_markov_hybrid_refused_with_reason(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(DFLASH_MARKOV_HYBRID))
    with pytest.raises(ValueError, match="Markov head"):
        load_dflash(str(tmp_path))


# ---------------------------------------------------------------- tap sanity probe

class _TinyLayer:
    def __init__(self, h):
        import mlx.nn as nn
        self.lin = nn.Linear(h, h, bias=False)

    def __call__(self, x, mask=None, cache=None):
        return x + self.lin(x)


class _TinyLM:
    """Minimal mlx-lm-shaped dense model: model.{embed_tokens,layers,norm} + lm_head."""

    def __init__(self, model_type="tinydense", embed_shift=0.0, **args):
        import mlx.nn as nn
        h, vocab = 16, 32
        self.model = SimpleNamespace(
            embed_tokens=nn.Embedding(vocab, h),
            layers=[_TinyLayer(h) for _ in range(2)],
            norm=nn.RMSNorm(h),
        )
        self.lm_head = nn.Linear(h, vocab, bias=False)
        self.args = SimpleNamespace(model_type=model_type, **args)
        self._shift = embed_shift

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, ids, cache=None):
        h = self.model.embed_tokens(ids) + self._shift
        for layer, c in zip(self.model.layers, cache or [None, None]):
            h = layer(h, None, c)
        return self.lm_head(self.model.norm(h))


def test_verify_tap_passes_for_plain_dense_model():
    Target(_TinyLM(), tokenizer=None).verify_tap()  # must not raise


def test_verify_tap_catches_forward_divergence():
    # a family whose forward does more than embed→layers→norm (gemma-style embedding
    # treatment; an additive shift here because the tiny fake is positively homogeneous,
    # so a multiplicative scale would cancel through its RMSNorm)
    with pytest.raises(ValueError, match="diverges"):
        Target(_TinyLM(embed_shift=1.0), tokenizer=None).verify_tap()


def test_verify_tap_refuses_windowed_attention():
    model = _TinyLM(model_type="windowed_fam",
                    layer_types=["sliding_attention", "full_attention"])
    with pytest.raises(ValueError, match="windowed"):
        Target(model, tokenizer=None).verify_tap()


def test_verify_tap_refuses_sliding_window_flag():
    model = _TinyLM(model_type="sw_fam", use_sliding_window=True, sliding_window=2048)
    with pytest.raises(ValueError, match="windowed"):
        Target(model, tokenizer=None).verify_tap()
