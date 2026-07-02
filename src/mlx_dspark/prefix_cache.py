"""Prefix KV caching for the OpenAI server — skip re-prefilling a shared conversation prefix.

A multi-turn request's prompt is (almost) the previous prompt + the assistant's reply + the new
user turn, so the bulk of it was already computed last turn. This keeps the target KV cache (and,
for DSpark, the drafter's context cache) from the most recent sequence and, on the next request,
reuses the longest common prefix: trim the caches back to it and prefill only the new suffix.

Scope (chosen for correctness):
  * **Dense / full-attention targets only.** Reuse trims the KV cache back to an arbitrary earlier
    position; that's exact for a plain ``KVCache`` (Qwen3) but *not* for the rotating/sliding-window
    caches Gemma-4 uses — those are gated off (the engine falls back to a fresh prefill).
  * **dspark + baseline modes.** DFlash's drafter uses its own sliding-window cache, so it isn't
    reused here.
  * **Single conversation.** One persistent entry — ideal for a single-user local server (LM Studio
    style). Not a multi-tenant KV pool.

Losslessness: reuse is lossless to the same standard as the rest of mlx-dspark — the output is a
valid decoding of the target, differing from a cold run only at logit-margin≈0 ties (fp
nondeterminism between chunked and single-pass prefill). The entry is invalidated on *any* error, so
a partially-advanced cache can never desync from the token record it claims to represent.

Optional **L2 SSD spill**: when the in-RAM cache exceeds a byte budget it is serialized to a
directory (target KV via mlx-lm's ``save_prompt_cache``, drafter ctx as safetensors) and dropped from
RAM, reloaded on the next reuse — for very long contexts, and to survive idle memory pressure.
"""

from __future__ import annotations

import glob
import json
import os

import mlx.core as mx

from .model import CtxCache


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _trim_target(cache, to_len: int) -> None:
    for c in cache:
        off = getattr(c, "offset", None)
        if off is None or not hasattr(c, "trim"):
            continue
        n = int(off) - to_len
        if n > 0:
            c.trim(n)


def target_cache_reusable(cache) -> bool:
    """True only if every layer cache retains all positions and trims to any length — i.e. a
    plain KVCache. Rotating/sliding-window/quantized caches are rejected (unsafe to roll back)."""
    return all(type(c).__name__ == "KVCache" and hasattr(c, "trim") and hasattr(c, "offset")
               for c in cache)


def _cache_ram_bytes(cache) -> int:
    total = 0
    for c in cache:
        st = getattr(c, "state", None)
        if isinstance(st, (list, tuple)):
            for a in st:
                total += getattr(a, "nbytes", 0)
    return total


class PrefixCache:
    def __init__(self, make_cache, make_ctx=None, *, min_reuse: int = 16,
                 l2_dir: str | None = None, max_ram_bytes: int = 0):
        self.make_cache = make_cache          # () -> list[target layer cache]
        self.make_ctx = make_ctx              # () -> list[CtxCache] | None (None for baseline)
        self.min_reuse = max(1, min_reuse)
        self.l2_dir = l2_dir
        self.max_ram_bytes = max_ram_bytes    # 0 = never spill (pure in-memory)
        self.hits = 0
        self.reused_tokens = 0
        self._tokens: list[int] = []          # tokens the stored cache holds KV for
        self._cache = None                    # in-RAM target cache (None if empty/spilled)
        self._ctx = None
        self._spilled = False
        self._valid = False
        if l2_dir:
            os.makedirs(l2_dir, exist_ok=True)

    # -- public API (engine calls these under its generation lock) --
    def acquire(self, prompt_ids: list[int]):
        """Return ``(cache, ctx, reuse_len)`` for this request — either the reused-and-trimmed
        persistent caches, or fresh ones."""
        reuse_len = self._reuse_len(prompt_ids)
        if reuse_len >= self.min_reuse:
            cache, ctx = self._materialize()
            if cache is not None:
                _trim_target(cache, reuse_len)
                if ctx is not None:
                    for c in ctx:
                        c.trim_to(reuse_len)
                self._valid = False           # in flight; store() re-validates
                self.hits += 1
                self.reused_tokens += reuse_len
                return cache, ctx, reuse_len
        self._valid = False
        return self.make_cache(), (self.make_ctx() if self.make_ctx else None), 0

    def store(self, cache, ctx, prompt_ids: list[int], token_ids: list[int]) -> None:
        # the cache holds KV for the prompt + every generated token EXCEPT the last (that one is
        # the pending token, not yet fed through the target) — see the generate loops.
        self._cache = cache
        self._ctx = ctx
        self._tokens = list(prompt_ids) + list(token_ids[:-1])
        self._spilled = False
        self._valid = True
        self._maybe_spill()

    def reset(self) -> None:
        self._cache = self._ctx = None
        self._tokens = []
        self._spilled = False
        self._valid = False
        self._clear_spill_files()

    def info(self) -> dict:
        return {"enabled": True, "cached_tokens": len(self._tokens) if self._valid else 0,
                "hits": self.hits, "reused_tokens": self.reused_tokens,
                "l2": bool(self.l2_dir), "spilled": self._spilled}

    # -- internals --
    def _reuse_len(self, prompt_ids: list[int]) -> int:
        if not self._valid or not self._tokens:
            return 0
        lcp = _lcp(self._tokens, prompt_ids)
        # keep at least one token to prefill (need logits for the next token)
        return max(0, min(lcp, len(self._tokens), len(prompt_ids) - 1))

    def _materialize(self):
        if self._cache is not None:
            return self._cache, self._ctx
        if self._spilled and self.l2_dir:
            return self._load_spill()
        return None, None

    def _maybe_spill(self) -> None:
        if self.max_ram_bytes <= 0 or not self.l2_dir or self._cache is None:
            return
        if _cache_ram_bytes(self._cache) <= self.max_ram_bytes:
            return
        self._save_spill()
        self._cache = self._ctx = None        # free RAM; reload on next reuse
        self._spilled = True

    # -- L2 SSD spill --
    def _target_path(self):
        return os.path.join(self.l2_dir, "target_cache.safetensors")

    def _ctx_path(self):
        return os.path.join(self.l2_dir, "ctx_cache.safetensors")

    def _save_spill(self) -> None:
        from mlx_lm.models.cache import save_prompt_cache
        save_prompt_cache(self._target_path(), self._cache,
                          metadata={"tokens": json.dumps(self._tokens)})
        if self._ctx is not None:
            arrs = {}
            for i, c in enumerate(self._ctx):
                if c.k is not None:
                    arrs[f"{i}.k"] = c.k
                    arrs[f"{i}.v"] = c.v
            mx.save_safetensors(self._ctx_path(), arrs)

    def _load_spill(self):
        from mlx_lm.models.cache import load_prompt_cache
        cache, _meta = load_prompt_cache(self._target_path(), return_metadata=True)
        ctx = None
        if self.make_ctx is not None and os.path.exists(self._ctx_path()):
            ctx = self.make_ctx()
            arrs = mx.load(self._ctx_path())
            for i, c in enumerate(ctx):
                if f"{i}.k" in arrs:
                    c.k, c.v = arrs[f"{i}.k"], arrs[f"{i}.v"]
        self._cache, self._ctx = cache, ctx   # back in RAM
        self._spilled = False
        return cache, ctx

    def _clear_spill_files(self) -> None:
        if not self.l2_dir:
            return
        for p in glob.glob(os.path.join(self.l2_dir, "*_cache.safetensors")):
            try:
                os.remove(p)
            except OSError:
                pass
