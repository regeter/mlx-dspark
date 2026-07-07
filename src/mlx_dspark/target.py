"""Family-aware target wrapper: KV cache + a hidden-state tap at given layers.

- gemma4 (mlx-vlm): uses the built-in ``capture_layer_ids`` / ``hidden_sink`` hook.
- qwen3  (mlx-lm):  no hook exists, so we replicate the model's forward loop and
  capture the residual stream after the tapped layers.

Both expose: make_cache(), run(ids, cache, tap)->(logits, fused_hidden), and
plain(ids, cache)->logits (no capture, for the greedy baseline).
"""

from __future__ import annotations

import mlx.core as mx


class Target:
    def __init__(self, model, tokenizer, *, kv_bits: int | None = None,
                 kv_group_size: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        # mlx-vlm models expose .language_model; mlx-lm models expose .model + (lm_head|tied)
        self.is_vlm = hasattr(model, "language_model")
        self.family = ("gemma4" if self.is_vlm else
                       getattr(getattr(model, "args", None), "model_type", "qwen3"))
        if kv_bits and self.is_vlm:
            raise ValueError("--kv-bits is supported for mlx-lm text targets only "
                             "(the mlx-vlm/gemma-4 cache layout is managed by mlx-vlm)")
        self.kv_bits = int(kv_bits) if kv_bits else None
        self.kv_group_size = int(kv_group_size)
        if not self.is_vlm:
            # mlx-lm convention: tied models simply don't define lm_head (more reliable
            # than args.tie_word_embeddings, which some families omit)
            self._tied = not hasattr(model, "lm_head")

    # -- cache --
    def make_cache(self):
        if self.is_vlm:
            return self.model.language_model.make_cache()
        if self.kv_bits:
            # Quantized KV from token 0: trimmable (spec rollback + prefix reuse work
            # unchanged), halves-or-quarters the KV bandwidth bill on long contexts.
            # Output is the greedy decoding of the KV-quantized target (a quality knob of
            # the same class as target quantization, not a spec-decoding approximation).
            from mlx_lm.models.cache import QuantizedKVCache
            return [QuantizedKVCache(self.kv_group_size, self.kv_bits)
                    for _ in self.model.layers]
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self.model)

    # -- forward with hidden-state tap --
    def run(self, ids: mx.array, cache, tap: list[int]):
        """ids [1,L] -> (logits [1,L,V], fused_hidden [1,L,n_tap*H])."""
        if self.is_vlm:
            out = self.model.language_model(inputs=ids, cache=cache, capture_layer_ids=tap)
            return out.logits, mx.concatenate(out.hidden_states, axis=-1)
        return self._run_mlxlm(ids, cache, tap)

    def _run_mlxlm(self, ids, cache, tap):
        from mlx_lm.models.base import create_attention_mask

        mm = self.model.model
        tapset = set(tap)
        h = mm.embed_tokens(ids)
        mask = create_attention_mask(h, cache[0])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask, c)
            if i in tapset:
                captured.append(h)
        hn = mm.norm(h)
        if self._tied:
            logits = mm.embed_tokens.as_linear(hn)
        else:
            logits = self.model.lm_head(hn)
        return logits, mx.concatenate(captured, axis=-1)

    # -- plain forward (no capture) for the greedy baseline --
    def plain(self, ids: mx.array, cache):
        if self.is_vlm:
            return self.model.language_model(inputs=ids, cache=cache).logits
        return self.model(ids, cache=cache)

    # -- tap sanity probe (drafter modes) --
    def verify_tap(self) -> None:
        """Prove the manual mlx-lm tap is faithful for THIS model, or fail loudly.

        ``_run_mlxlm`` replicates the plain dense forward (embed → layers → norm → head).
        A family whose forward does more — embedding scaling (gemma), per-layer
        sliding-window masks, extra streams — would draft from a silently-wrong hidden
        stream, which wastes far more user time than an error. Two checks:
        (1) structural: refuse windowed/alternating attention (a short probe can't
        exercise a window, so it must be refused, not probed); (2) numeric: the
        replicated loop must reproduce the model's own logits on a tiny input
        (identical ops/widths → should match bit-for-bit; 1e-3 is generous headroom).
        Costs two 4-token forwards, once per load. VLM targets use the native capture
        hook — nothing replicated, nothing to verify."""
        if self.is_vlm:
            return
        args = getattr(self.model, "args", None)
        mt = getattr(args, "model_type", "?")
        layer_types = getattr(args, "layer_types", None) or []
        windowed = (any(t != "full_attention" for t in layer_types)
                    or (getattr(args, "use_sliding_window", False)
                        and getattr(args, "sliding_window", None)))
        if windowed:
            raise ValueError(
                f"hidden-state tap unsupported for model_type {mt!r}: it uses "
                f"windowed/alternating attention layers, which the generic tap's single "
                f"causal mask would get wrong past the window. baseline/lookup modes still "
                f"work; for drafter support open an issue: "
                f"https://github.com/ARahim3/mlx-dspark/issues"
            )
        ids = mx.array([[1, 2, 3, 4]])
        try:
            ref = self.plain(ids, self.make_cache())
            got, _ = self._run_mlxlm(ids, self.make_cache(), [0])
            diff = float(mx.abs(ref - got).max())
        except Exception as e:
            raise ValueError(
                f"hidden-state tap unsupported for model_type {mt!r}: the generic forward "
                f"loop failed ({e.__class__.__name__}: {e}). baseline/lookup modes still work."
            ) from e
        if diff > 1e-3:
            raise ValueError(
                f"hidden-state tap unsupported for model_type {mt!r}: the replicated forward "
                f"diverges from the model's own (max |Δlogit| = {diff:.4g}) — this family's "
                f"forward does more than embed→layers→norm. baseline/lookup modes still work."
            )
