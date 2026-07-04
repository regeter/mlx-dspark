"""Continuous batching for mlx-dspark — run B requests through the target in one
forward so they share a single weight-read per step (verify cost is memory-bound; B
sequences amortize it). This is the "cheap-verify" regime the paper gets from batched GPU
serving, brought to a Mac.

**Model contract (why this is general, not qwen-specific).** The batched forward is the
*generic* mlx-lm dense-model loop — ``model.model.{embed_tokens, layers, norm}`` with
``layer(h, mask, cache)`` and attention that reads ``cache.offset`` for RoPE — the exact
shape ``target.py::_run_mlxlm`` already taps. Any standard mlx-lm dense target (Qwen3-4B/8B/
14B, Llama, Mistral, Phi, …) satisfies it; :func:`batchable` gates on it so a *new dense model*
works automatically and anything unproven (gemma-4 vlm + rotating/sliding cache) cleanly falls
back to the serialized engine instead of decoding wrong. The only batching-specific piece is
:class:`BatchCache`, which duck-types the KV-cache interface the layers call — so it slots into
any of those models unchanged.

Batched *spec* decoding needs each row to roll its KV back by a different amount every round
(rows accept different numbers of drafted tokens). mlx-lm's own ``BatchKVCache`` shares one
write cursor and can only trim uniformly, so this module carries its own **left-aligned,
per-row-offset** cache: row b's real tokens live in columns ``0..offset[b]``, writes scatter to
each row's own offset, and ``trim`` is pure per-row metadata (the same O(1) trick the single-seq
``KVCache`` uses). Per-row RoPE rides ``mx.fast.rope``'s array-offset support (also model-agnostic).
"""

from __future__ import annotations

import time

import mlx.core as mx

from .generate import (
    GenResult,
    _prefill_plain,
    _Streamer,
    eos_token_ids,
)

STEP = 256  # KV buffer grows in chunks of this many columns (like mlx-lm's caches)


# --------------------------------------------------------------------------- capability


def batchable(target) -> bool:
    """True iff this target supports the batched path: a dense mlx-lm model whose layers use
    the standard ``KVCache`` (so per-row offsets + our mask are correct). VLM/gemma-4 (rotating
    / sliding cache, mlx-vlm wrapper) returns False -> caller falls back to the serialized
    engine. Extending to a new family means teaching the forward loop + cache, not a silent break.
    """
    if getattr(target, "is_vlm", False):
        return False
    try:
        cache = target.make_cache()
    except Exception:  # noqa: BLE001
        return False
    return bool(cache) and all(type(c).__name__ == "KVCache" for c in cache)


# --------------------------------------------------------------------------- batched cache


class BatchCache:
    """Left-aligned per-row KV cache for one layer of a batched decode.

    ``keys``/``values`` are ``[B, H, Lbuf, D]``; row b's real content occupies columns
    ``0..offsets[b]`` (contiguous, no gaps — trimmed tail garbage is overwritten by the next
    write). ``offsets[b]`` is both the row's logical length and the RoPE start position of its
    next token. Duck-types what mlx-lm attention calls: ``.offset`` (array, for rope) and
    ``.update_and_fetch``. Masking is supplied externally (see :func:`build_batch_mask`) because
    the per-row causal+padding mask depends only on the offsets the caller already holds.
    """

    def __init__(self, keys: mx.array, values: mx.array, offsets: list[int]):
        self.keys = keys
        self.values = values
        self.offsets = list(offsets)  # python list: cheap per-row indexing for writes/trim/mask

    # -- construction --
    @classmethod
    def from_rows(cls, kv_pairs: list[tuple[mx.array, mx.array]]) -> "BatchCache":
        """Merge per-row single-seq (keys, values) — each ``[1, H, len_b, D]`` — into one
        left-aligned batched buffer (row b at columns ``0..len_b``)."""
        lens = [k.shape[2] for k, _ in kv_pairs]
        B = len(kv_pairs)
        H = kv_pairs[0][0].shape[1]
        Dk = kv_pairs[0][0].shape[3]
        Dv = kv_pairs[0][1].shape[3]
        dt = kv_pairs[0][0].dtype
        Lbuf = _round_step(max(lens))
        keys = mx.zeros((B, H, Lbuf, Dk), dtype=dt)
        values = mx.zeros((B, H, Lbuf, Dv), dtype=dt)
        for b, (k, v) in enumerate(kv_pairs):
            n = lens[b]
            keys[b : b + 1, :, :n, :] = k
            values[b : b + 1, :, :n, :] = v
        return cls(keys, values, lens)

    # -- interface the attention layers use --
    @property
    def offset(self) -> mx.array:
        return mx.array(self.offsets, dtype=mx.int32)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Write each row's ``T`` new tokens at its own offset, advance offsets, and return the
        full ``[B, H, Lcur, D]`` key/value tensors (``Lcur = max(new offsets)``). Rows shorter
        than ``Lcur`` carry stale/other-row content past their offset — the mask hides it."""
        B, H, T, _ = keys.shape
        need = max(o + T for o in self.offsets)
        if need > self.keys.shape[2]:
            self._grow(_round_step(need))
        for b in range(B):
            o = self.offsets[b]
            self.keys[b : b + 1, :, o : o + T, :] = keys[b : b + 1]
            self.values[b : b + 1, :, o : o + T, :] = values[b : b + 1]
            self.offsets[b] = o + T
        Lcur = max(self.offsets)
        return self.keys[:, :, :Lcur, :], self.values[:, :, :Lcur, :]

    def _grow(self, new_lbuf: int) -> None:
        B, H, _, Dk = self.keys.shape
        Dv = self.values.shape[3]
        nk = mx.zeros((B, H, new_lbuf, Dk), dtype=self.keys.dtype)
        nv = mx.zeros((B, H, new_lbuf, Dv), dtype=self.values.dtype)
        L = self.keys.shape[2]
        nk[:, :, :L, :] = self.keys
        nv[:, :, :L, :] = self.values
        self.keys, self.values = nk, nv

    def trim(self, ns: list[int]) -> None:
        """Roll each row back by ``ns[b]`` tokens — pure metadata (the trimmed tail is
        overwritten by the next write). ``ns[b] == 0`` leaves row b untouched."""
        for b, n in enumerate(ns):
            self.offsets[b] = max(0, self.offsets[b] - n)


def _round_step(n: int) -> int:
    return ((n + STEP - 1) // STEP) * STEP


def build_batch_mask(offsets: list[int], T: int) -> mx.array:
    """Boolean attention mask ``[B, 1, T, Lcur]`` (Lcur = max(offsets)+T): row b query position
    i (absolute offset[b]+i) attends key column j iff ``j <= offset[b] + i`` — causal within the
    new block *and* padding-safe against shorter rows / stale tail. Broadcasts over heads."""
    off = mx.array(offsets, dtype=mx.int32)[:, None, None, None]  # [B,1,1,1]
    Lcur = max(offsets) + T
    i = mx.arange(T, dtype=mx.int32)[None, None, :, None]         # [1,1,T,1]
    j = mx.arange(Lcur, dtype=mx.int32)[None, None, None, :]      # [1,1,1,Lcur]
    return j <= (off + i)


# --------------------------------------------------------------------------- batched forward


def batched_forward(target, ids: mx.array, caches: list[BatchCache], tap: list[int] | None = None,
                    mask=None):
    """One batched forward through a dense mlx-lm target with per-row :class:`BatchCache`.

    ``ids`` is ``[B, T]`` (T=1 for baseline decode, cap+1 for verify). Returns
    ``(logits [B, T, V], fused [B, T, n_tap*H] | None)``. The generic layer loop mirrors
    ``target.py::_run_mlxlm`` (embed -> layers(h, mask, cache) -> norm -> lm_head/tied), so it is
    not tied to any one model; ``tap`` captures the residual stream after the given layer ids for
    the DSpark drafter (None = plain forward, the baseline path)."""
    mm = target.model.model
    if mask is None:
        mask = build_batch_mask(caches[0].offsets, ids.shape[1])
    h = mm.embed_tokens(ids)
    tapset = set(tap or [])
    captured = []
    for i, (layer, c) in enumerate(zip(mm.layers, caches)):
        h = layer(h, mask, c)
        if i in tapset:
            captured.append(h)
    hn = mm.norm(h)
    logits = mm.embed_tokens.as_linear(hn) if target._tied else target.model.lm_head(hn)
    fused = mx.concatenate(captured, axis=-1) if captured else None
    return logits, fused


def _prefill_rows(target, prompts_ids: list[list[int]], tap: list[int] | None = None):
    """Prefill each row's prompt individually (one-at-a-time bounds memory and keeps the code
    on the existing single-seq prefill), returning per-row (last_logits, cache, fused|None)."""
    rows = []
    for ids in prompts_ids:
        cache = target.make_cache()
        if tap is None:
            logits = _prefill_plain(target, ids, cache)
            fused = None
        else:
            from .generate import _prefill_tapped

            logits, fused = _prefill_tapped(target, ids, cache, tap)
        rows.append((logits, cache, fused))
    return rows


def _merge_row_caches(row_caches: list[list]) -> list[BatchCache]:
    """Per-layer BatchCache list from a list of per-row single-seq cache lists."""
    n_layers = len(row_caches[0])
    out = []
    for l in range(n_layers):
        pairs = [(c[l].keys[..., : c[l].offset, :], c[l].values[..., : c[l].offset, :])
                 for c in row_caches]
        out.append(BatchCache.from_rows(pairs))
    return out


def _sample_rows(logits_last: mx.array, temperature: float, top_p: float, top_k: int) -> mx.array:
    """Next token per row from ``[B, V]`` logits — argmax (temp 0) or truncated sample."""
    if temperature > 0.0:
        from .sampling import sample_probs, truncate_probs

        probs = truncate_probs(mx.softmax(logits_last / temperature, axis=-1), top_p, top_k)
        return sample_probs(probs)
    return mx.argmax(logits_last, axis=-1)


# --------------------------------------------------------------------------- per-row bookkeeping


def _as_list(x, B):
    return list(x) if isinstance(x, (list, tuple)) else [x] * B


class _RowTracker:
    """Per-row state for a batched generation: streaming detok + stop strings + eos + per-row
    ``max_new_tokens`` + result assembly. One shared abstraction so the batched baseline and spec
    loops both get streaming / stop-sequences / independent per-row lengths (what a server needs)
    without duplicating the single-seq loop's :class:`~mlx_dspark.generate._Streamer` handling.

    ``first_tokens`` are the prefill-sampled tokens (counted as each row's token 1, and streamed).
    ``commit(b, toks)`` appends a round's committed tokens for row b; ``finished[b]`` flips on
    eos / length / stop-string. ``results(secs)`` returns one :class:`GenResult` per row."""

    def __init__(self, tokenizer, first_tokens, *, max_new_tokens, on_texts=None, stops=None):
        self.tok = tokenizer
        self.eos = eos_token_ids(tokenizer)
        B = len(first_tokens)
        self.B = B
        self.mnt = _as_list(max_new_tokens, B)
        ot = on_texts or [None] * B
        sp = stops or [None] * B
        self.streamers = [_Streamer(tokenizer, self.eos, ot[b], sp[b]) for b in range(B)]
        self.out_ids = [[int(first_tokens[b])] for b in range(B)]
        self.accept_lengths: list[list[int]] = [[] for _ in range(B)]
        self.finished = [False] * B
        for b in range(B):
            self.streamers[b].update(self.out_ids[b])
            self._check(b)

    def _check(self, b):
        ids = self.out_ids[b]
        if ids[-1] in self.eos or len(ids) >= self.mnt[b] or self.streamers[b].stopped:
            self.finished[b] = True

    def commit(self, b, tokens, accept_len=None):
        if self.finished[b]:
            return
        if accept_len is not None:
            self.accept_lengths[b].append(accept_len)
        for tok in tokens:
            self.out_ids[b].append(tok)
            if tok in self.eos:
                break
        self.streamers[b].update(self.out_ids[b])
        self._check(b)

    def all_done(self):
        return all(self.finished)

    def results(self, secs):
        out = []
        for b in range(self.B):
            self.streamers[b].flush()
            ids = self.out_ids[b]
            stopped = self.streamers[b].stopped
            reason = "stop" if (ids[-1] in self.eos or stopped) else "length"
            text = (self.streamers[b].text if stopped
                    else self.tok.decode([t for t in ids if t not in self.eos]))
            spec = bool(self.accept_lengths[b])
            al = self.accept_lengths[b] if spec else [1] * len(ids)
            out.append(GenResult(
                text=text, token_ids=ids, num_tokens=len(ids), num_rounds=len(al),
                accept_lengths=al, target_forwards=(len(al) + 1) if spec else len(ids),
                seconds=secs, finish_reason=reason,
            ))
        return out


# --------------------------------------------------------------------------- baseline (M1)


def batch_generate_baseline(
    target,
    tokenizer,
    prompts_ids: list[list[int]],
    *,
    max_new_tokens=128,                    # int, or a per-row list[int]
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    on_texts=None,                          # per-row on_text callbacks (streaming)
    stops=None,                             # per-row stop-string lists
    seed: int | None = None,
) -> list[GenResult]:
    """Static batched plain decode of B prompts (no drafter) — the batched baseline. Every row
    advances one token per step through one shared forward; a finished row keeps stepping (its
    output ignored) until all rows finish. Row isolation is the losslessness contract: B=N
    produces the same per-row ids as B=1 at decode width 1 (the batched forward is bit-identical
    there — see NOTES). Returns one :class:`GenResult` per row; batching is pure throughput."""
    if seed is not None:
        mx.random.seed(seed)
    B = len(prompts_ids)
    t0 = time.time()

    rows = _prefill_rows(target, prompts_ids)
    caches = _merge_row_caches([c for _, c, _ in rows])
    first = _sample_rows(mx.stack([lg[0, -1] for lg, _, _ in rows]), temperature, top_p, top_k)
    mx.eval(first)
    trk = _RowTracker(tokenizer, [int(x) for x in first.tolist()],
                      max_new_tokens=max_new_tokens, on_texts=on_texts, stops=stops)
    cur = first.reshape(B, 1)

    while not trk.all_done():
        logits, _ = batched_forward(target, cur, caches)
        ynext = _sample_rows(logits[:, -1], temperature, top_p, top_k)
        mx.eval(ynext)
        toks = ynext.tolist()
        for b in range(B):
            trk.commit(b, [int(toks[b])])
        cur = ynext.reshape(B, 1)

    return trk.results(time.time() - t0)


# --------------------------------------------------------------------------- spec (M2, Stage A)


class _BatchCtx:
    """Batched context K/V for one drafter layer — duck-types the ``.k`` / ``.v`` that
    :meth:`DSparkAttention.attend` reads, so ``drafter.backbone`` runs batched unchanged."""
    __slots__ = ("k", "v")

    def __init__(self, k, v):
        self.k = k
        self.v = v


def _batched_ctx(ctx_rows: list[list]) -> tuple[list[_BatchCtx], list[int]]:
    """Pad the per-row drafter :class:`CtxCache`s into batched ``[B, n_kv, Lctx_max, D]`` buffers
    (one :class:`_BatchCtx` per layer) + the per-row context lengths. ``ctx_rows`` is a list over
    rows of the per-layer cache list; context length is identical across layers (update_context
    appends to every layer), so one length vector serves the block mask and the RoPE offset."""
    B = len(ctx_rows)
    n_layers = len(ctx_rows[0])
    lens = [ctx_rows[b][0].length for b in range(B)]
    Lmax = max(lens)
    out = []
    for l in range(n_layers):
        c0 = ctx_rows[0][l]
        n_kv, Dk, Dv = c0.k.shape[1], c0.k.shape[3], c0.v.shape[3]
        bk = mx.zeros((B, n_kv, Lmax, Dk), dtype=c0.k.dtype)
        bv = mx.zeros((B, n_kv, Lmax, Dv), dtype=c0.v.dtype)
        for b in range(B):
            cb = ctx_rows[b][l]
            bk[b : b + 1, :, : lens[b], :] = cb.k
            bv[b : b + 1, :, : lens[b], :] = cb.v
        out.append(_BatchCtx(bk, bv))
    return out, lens


def _ctx_block_mask(lens: list[int], k: int) -> mx.array:
    """Boolean ``[B, 1, k, Lmax+k]`` mask for the batched block attention: each block position
    attends its row's valid context (columns ``< lens[b]``) plus every block position (columns
    ``>= Lmax``, bidirectional) — hiding shorter rows' context padding. Matches the single-seq
    full-attention exactly per row."""
    B = len(lens)
    Lmax = max(lens)
    lens_a = mx.array(lens)[:, None, None, None]                 # [B,1,1,1]
    j = mx.arange(Lmax + k)[None, None, None, :]                 # [1,1,1,Lmax+k]
    m = (j >= Lmax) | (j < lens_a)                              # block region OR valid context
    return mx.broadcast_to(m, (B, 1, k, Lmax + k))


def _batched_sample_block(drafter, base_logits: mx.array, first_prev: list[int]) -> mx.array:
    """Batched greedy DSpark block sampling: ``base_logits [B, cap, V]``, ``first_prev [B]`` ->
    ``[B, cap]``. The Markov head is sequential over the cap positions (position i's bias depends
    on the token at i-1) but vectorized across rows."""
    B, cap, _ = base_logits.shape
    if drafter.markov_head is None:
        return mx.argmax(base_logits, axis=-1)
    prev = mx.array(first_prev)                                  # [B]
    toks = []
    for i in range(cap):
        step = base_logits[:, i, :] + drafter.markov_head.step_bias(prev)   # [B, V]
        nxt = mx.argmax(step, axis=-1)                          # [B]
        toks.append(nxt)
        prev = nxt
    return mx.stack(toks, axis=1)                               # [B, cap]


def batch_spec_generate(
    target,
    tokenizer,
    drafter,
    prompts_ids: list[list[int]],
    *,
    max_new_tokens=128,                    # int, or a per-row list[int]
    max_draft_tokens: int = 2,
    batched_drafter: bool = True,           # Stage B: batch the drafter too (False = Stage A)
    on_texts=None,                          # per-row on_text callbacks (streaming)
    stops=None,                             # per-row stop-string lists
    seed: int | None = None,
) -> list[GenResult]:
    """Static batched **DSpark** speculative decode (greedy) — Stage A: the target *verify*
    is batched (all rows share one weight-read), the drafter runs sequentially per row.

    Every row drafts a fixed ``cap = max_draft_tokens`` tokens, so the verify block is a uniform
    ``[B, cap+1]``; rows accept different prefixes and each rolls its own KV back by ``cap - n``
    via the per-row :meth:`BatchCache.trim`. Output per row is the target's greedy decoding — B=1
    is bit-exact vs single-seq; at B>1 it's greedy-correct up to the target's batch-dependent
    quantized-matmul rounding (see NOTES). Returns one GenResult per row."""
    if seed is not None:
        mx.random.seed(seed)
    cfg = drafter.config
    tap = list(cfg.target_layer_ids)
    k = int(cfg.block_size)
    mask_id = int(cfg.mask_token_id)
    cap = max(1, min(int(max_draft_tokens), k))
    B = len(prompts_ids)
    t0 = time.time()

    # --- per-row prefill (seeds each row's drafter ctx) + merge target caches into a batch ---
    from .generate import _prefill_tapped

    row_caches, ctx_caches, n_cached, pending = [], [], [], []
    for ids in prompts_ids:
        cache = target.make_cache()
        ctx = drafter.make_ctx_cache()
        logits, _ = _prefill_tapped(target, ids, cache, tap, drafter=drafter, ctx_caches=ctx)
        row_caches.append(cache)
        ctx_caches.append(ctx)
        n_cached.append(len(ids))
        pending.append(int(mx.argmax(logits[0, -1]).item()))
    caches = _merge_row_caches(row_caches)
    trk = _RowTracker(tokenizer, pending, max_new_tokens=max_new_tokens,
                      on_texts=on_texts, stops=stops)

    while not trk.all_done():
        # ---- 1. draft ----
        if batched_drafter:
            # Stage B: one batched backbone forward over [B, k, H]. The ragged part is the
            # per-row context — pad it to a rectangle + a mask that hides each row's padding, and
            # feed a per-row RoPE offset (rows sit at different context lengths). Reuses
            # drafter.backbone unchanged (attend now takes the mask).
            block_ids = mx.array([[pending[b]] + [mask_id] * (k - 1) for b in range(B)])  # [B,k]
            noise = drafter.embed(block_ids)                                    # [B, k, H]
            bctx, ctx_lens = _batched_ctx(ctx_caches)
            mask = _ctx_block_mask(ctx_lens, k)
            block_hidden = drafter.backbone(noise, mx.array(ctx_lens), bctx, mask)   # [B, k, H]
            base_logits = drafter.compute_logits(block_hidden[:, :cap, :])       # [B, cap, V]
            drafts_arr = _batched_sample_block(drafter, base_logits, pending)    # [B, cap]
            drafts = [drafts_arr[b] for b in range(B)]
        else:
            # Stage A: run the drafter sequentially per row (batched verify only).
            drafts = []
            for b in range(B):
                block_ids = [pending[b]] + [mask_id] * (k - 1)
                noise = drafter.embed(mx.array([block_ids]))
                block_hidden = drafter.backbone(noise, n_cached[b], ctx_caches[b])
                base_logits = drafter.compute_logits(block_hidden[:, :cap, :])[0]      # [cap, V]
                drafts.append(drafter.sample_block(base_logits, first_prev_token=pending[b]))

        # ---- 2. batched verify (the shared weight-read) ----
        verify_ids = mx.stack([mx.concatenate([mx.array([pending[b]], dtype=drafts[b].dtype),
                                               drafts[b]]) for b in range(B)])       # [B, cap+1]
        v_logits, v_fused = batched_forward(target, verify_ids, caches, tap)

        # ---- 3. accept per row (row-wise cumprod of the argmax match) ----
        tt = mx.argmax(v_logits, axis=-1)                                            # [B, cap+1]
        draft_stack = mx.stack(drafts)                                               # [B, cap]
        match = (draft_stack == tt[:, :cap]).astype(mx.int32)
        n_arr = mx.cumprod(match, axis=1).sum(axis=1)                                # [B]
        mx.eval(tt, n_arr, v_fused)
        n_list = n_arr.tolist()
        tt_list = tt.tolist()

        # ---- 4. commit + per-row trim + per-row drafter-ctx update ----
        trims = []
        for b in range(B):
            n = int(n_list[b])
            committed = [int(x) for x in drafts[b].tolist()[:n]] + [int(tt_list[b][n])]
            trims.append(cap - n)
            drafter.update_context(v_fused[b : b + 1, : n + 1, :],
                                   ctx_offset=n_cached[b], ctx_caches=ctx_caches[b])
            n_cached[b] += n + 1
            trk.commit(b, committed, accept_len=len(committed))
            pending[b] = trk.out_ids[b][-1]              # valid token even for finished rows
        for c in caches:                                  # per-row rollback, every layer
            c.trim(trims)

    return trk.results(time.time() - t0)
