"""Loaders for the target (Gemma-4 via mlx-vlm) and the DSpark drafter."""

from __future__ import annotations

import glob
import os

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .config import DSparkConfig
from .model import DSparkDrafter
from .target import Target

# The drafter must be paired with the *instruct* target it was trained against, at decent
# precision. Presets below; pick with load_pair("gemma4") or load_pair("qwen3").
PRESETS = {
    "gemma4": {
        "target": "mlx-community/gemma-4-12B-it-8bit",
        "drafter": "deepseek-ai/dspark_gemma4_12b_block7",
    },
    "qwen3": {
        "target": "mlx-community/Qwen3-4B-8bit",
        "drafter": "deepseek-ai/dspark_qwen3_4b_block7",
    },
}
DEFAULT_TARGET = PRESETS["gemma4"]["target"]
DEFAULT_DRAFTER = PRESETS["gemma4"]["drafter"]


def _resolve(repo_or_path: str) -> str:
    if os.path.isdir(repo_or_path):
        return repo_or_path
    return snapshot_download(repo_or_path)


def load_drafter(
    repo_or_path: str = DEFAULT_DRAFTER,
    *,
    quantize: bool = True,
    bits: int = 4,
    group_size: int = 64,
):
    """Return (drafter, config). Loads bf16 weights 1:1 by matching key names.

    The drafter is ~6.86 GB in bf16 and runs every speculative round, so by
    default it is quantized to 4-bit (~1.8 GB) — this is what makes spec
    decoding a net speedup on Apple Silicon. Output correctness is unaffected
    (the target verifies every token); only acceptance length may change.
    """
    path = _resolve(repo_or_path)
    config = DSparkConfig.from_json(os.path.join(path, "config.json"))
    drafter = DSparkDrafter(config)

    weights: dict[str, mx.array] = {}
    for st in glob.glob(os.path.join(path, "*.safetensors")):
        weights.update(mx.load(st))

    # Diagnose name mismatches before loading.
    model_keys = {k for k, _ in _flatten_params(drafter)}
    ckpt_keys = set(weights.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        print(f"[load_drafter] WARNING key mismatch:")
        if missing:
            print(f"  missing in checkpoint ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"  unexpected in checkpoint ({len(unexpected)}): {unexpected[:8]}")

    drafter.load_weights(list(weights.items()), strict=not (missing or unexpected))

    if quantize:
        # Quantize Linear/Embedding weights; norms/scalars stay full precision.
        nn.quantize(drafter, group_size=group_size, bits=bits)

    mx.eval(drafter.parameters())
    return drafter, config


def _flatten_params(module) -> list[tuple[str, mx.array]]:
    from mlx.utils import tree_flatten

    return tree_flatten(module.parameters())


def load_target(repo_or_path: str = DEFAULT_TARGET):
    """Return (Target, tokenizer). Routes text models (Qwen3) to mlx-lm and VLM/unified
    models (Gemma-4) to mlx-vlm, then wraps in a family-aware Target (hidden-state tap)."""
    import json

    path = _resolve(repo_or_path)
    model_type = ""
    cfg_path = os.path.join(path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            model_type = json.load(f).get("model_type", "")

    if "qwen3" in model_type and "moe" not in model_type:
        from mlx_lm import load as lm_load

        model, tokenizer = lm_load(path)
    else:
        from mlx_vlm import load as vlm_load

        model, processor = vlm_load(path)
        tokenizer = getattr(processor, "tokenizer", processor)
    return Target(model, tokenizer), tokenizer


def load_pair(family: str = "gemma4", *, target_bits: str | None = None):
    """Convenience: load (target, tokenizer, drafter, cfg) for a preset family."""
    p = PRESETS[family]
    target, tok = load_target(p["target"])
    drafter, cfg = load_drafter(p["drafter"])
    return target, tok, drafter, cfg
