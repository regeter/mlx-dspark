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
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # mlx-vlm models expose .language_model; mlx-lm models expose .model + (lm_head|tied)
        self.is_vlm = hasattr(model, "language_model")
        self.family = "gemma4" if self.is_vlm else "qwen3"
        if not self.is_vlm:
            self._tied = bool(getattr(getattr(model, "args", None), "tie_word_embeddings", False))

    # -- cache --
    def make_cache(self):
        if self.is_vlm:
            return self.model.language_model.make_cache()
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
