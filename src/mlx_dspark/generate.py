"""DSpark speculative decoding loop (greedy, batch=1) for Apple Silicon.

Per round:
  1. draft a block of K tokens from the parallel backbone + Markov head,
  2. verify them in one target forward,
  3. accept the matching prefix + 1 bonus token (so >=1 token/round always),
  4. trim the target KV cache and grow the fused-hidden context buffer.

Because the target verifies every token, the *output is exactly greedy target
decoding* regardless of drafter quality — drafter quality only shows up as the
acceptance length (tokens committed per target forward).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx

TAP = None  # set from drafter config at call time


@dataclass
class GenResult:
    text: str
    token_ids: list[int]
    num_tokens: int
    num_rounds: int
    accept_lengths: list[int]
    target_forwards: int
    seconds: float

    @property
    def mean_accept_len(self) -> float:
        return self.num_tokens / max(self.num_rounds, 1)

    @property
    def tokens_per_sec(self) -> float:
        return self.num_tokens / max(self.seconds, 1e-9)


def encode_prompt(tokenizer, prompt: str, use_chat: bool = True) -> list[int]:
    """Token ids for a user prompt, using the model's chat template when present.

    Gemma-4 uses `<|turn>` / `<channel|>` markers (NOT Gemma-3's `<start_of_turn>`),
    so the template must be applied via the tokenizer — hand-formatting breaks the
    instruct model. apply_chat_template may return list[int] or a BatchEncoding.
    """
    if use_chat and getattr(tokenizer, "chat_template", None):
        r = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        if isinstance(r, (list, tuple)):
            if r and isinstance(r[0], int):
                return list(r)
            if r and isinstance(r[0], (list, tuple)):
                return list(r[0])
        ii = None
        if hasattr(r, "__contains__") and "input_ids" in r:
            ii = r["input_ids"]
        elif hasattr(r, "input_ids"):
            ii = r.input_ids
        if ii is not None:
            ii = list(ii)
            return list(ii[0]) if ii and isinstance(ii[0], (list, tuple)) else ii
        if hasattr(r, "ids"):
            return list(r.ids)
    return list(tokenizer.encode(prompt))


def eos_token_ids(tokenizer) -> set[int]:
    """Collect stop-token ids: eos + Gemma turn-end markers (Gemma-4 uses <turn|>=106;
    note <end_of_turn> is the UNK id in Gemma-4, so it must be filtered out)."""
    ids: set[int] = set()
    e = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(e, int):
        ids.add(e)
    elif e:
        ids.update(int(x) for x in e)
    e1 = getattr(tokenizer, "eos_token_id", None)
    if isinstance(e1, int):
        ids.add(e1)
    unk = getattr(tokenizer, "unk_token_id", None)
    # Gemma-4 (<turn|>), Gemma-3 (<end_of_turn>), Qwen (<|im_end|>), raw eos
    for t in ("<turn|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>", "<eos>"):
        try:
            i = tokenizer.convert_tokens_to_ids(t)
        except Exception:
            continue
        if isinstance(i, int) and i >= 0 and i != unk:
            ids.add(i)
    return ids


def _pick(logits_row, temperature: float) -> int:
    """argmax (temperature 0) or a temperature sample (temperature > 0)."""
    if temperature > 0.0:
        return int(mx.random.categorical(logits_row / temperature).item())
    return int(mx.argmax(logits_row).item())


def greedy_generate(
    target_model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    seed: int | None = None,
    apply_chat_template: bool = True,
    on_text=None,
) -> GenResult:
    """Plain decoding of the target (no drafter, no hidden-state capture) — the fair
    'run the model normally' baseline. ``temperature`` 0 = greedy, > 0 = sampling (matches
    the spec path so a temp>0 A/B compares like-for-like). Streams via on_text."""
    if seed is not None:
        mx.random.seed(seed)
    eos_ids = eos_token_ids(tokenizer)
    ids = encode_prompt(tokenizer, prompt, use_chat=apply_chat_template)
    cache = target_model.make_cache()

    t0 = time.time()
    logits = target_model.plain(mx.array([ids]), cache)
    nxt = _pick(logits[0, -1], temperature)
    out_ids = [nxt]
    streamed = 0

    def _stream():
        nonlocal streamed
        if on_text is None:
            return
        disp = [t for t in out_ids if t not in eos_ids]
        full = tokenizer.decode(disp)
        if len(full) > streamed:
            on_text(full[streamed:])
            streamed = len(full)

    _stream()
    while len(out_ids) < max_new_tokens and nxt not in eos_ids:
        logits = target_model.plain(mx.array([[nxt]]), cache)
        nxt = _pick(logits[0, -1], temperature)
        out_ids.append(nxt)
        _stream()

    secs = time.time() - t0
    disp = [t for t in out_ids if t not in eos_ids]
    return GenResult(
        text=tokenizer.decode(disp),
        token_ids=out_ids,
        num_tokens=len(out_ids),
        num_rounds=len(out_ids),
        accept_lengths=[1] * len(out_ids),
        target_forwards=len(out_ids),
        seconds=secs,
    )


def dflash_generate(
    target_model,
    tokenizer,
    drafter,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    max_draft_tokens: int | None = None,
    temperature: float = 0.0,
    apply_chat_template: bool = True,
    seed: int | None = None,
    on_text=None,
) -> GenResult:
    """Speculative decoding with a **z-lab DFlash** (block-diffusion) drafter.

    DFlash differs from DSpark in two ways that matter to this loop:
      - it feeds ``[anchor] + (block-1) masks`` and reads logits at the **mask** positions
        (``logits_start=1``), i.e. predict-the-masks, not DSpark's anchor-as-position-0;
      - it has no own embed/lm_head — it reuses the target's (we ``bind`` once here).

    ``temperature == 0`` → greedy (exact argmax-match verify; output == greedy decoding up to
    fp ties). ``temperature > 0`` → speculative sampling (paper §2.1): drafts sampled from the
    block-diffusion proposal q, accepted w.p. ``min(1, p/q)`` vs the target p, residual-resampled
    on first reject — an exact sample from the target at temperature T (lossless).

    The backbone always drafts the full block width (it's trained at that width / block
    diffusion is bidirectional); ``max_draft_tokens`` only bounds how many drafted tokens
    are *verified* per round. ``None`` = verify the whole block (DFlash's native operating
    point — best on structured content; on open chat a short cap is faster).
    """
    if seed is not None:
        mx.random.seed(seed)
    if getattr(drafter, "embed_tokens", None) is None:
        drafter.bind(target_model.model)

    cfg = drafter.config
    tap = list(cfg.target_layer_ids)
    bs = int(cfg.block_size)
    mask_id = int(cfg.mask_token_id)
    kdraft = bs - 1
    cap = kdraft if max_draft_tokens is None else max(1, min(max_draft_tokens, kdraft))
    eos_ids = eos_token_ids(tokenizer)

    ids = encode_prompt(tokenizer, prompt, use_chat=apply_chat_template)
    cache = _make_target_cache(target_model)
    dcache = drafter.make_cache()                      # persistent draft ctx cache
    t0 = time.time()

    # prefill
    logits, fused = _run_target(target_model, mx.array([ids]), cache, tap)
    pending_ctx = fused                                # fused hidden appended to draft ctx next round
    pending = _pick(logits[0, -1], temperature)
    out_ids: list[int] = [pending]
    accept_lengths: list[int] = []
    target_forwards = 1
    streamed = 0

    def _stream():
        nonlocal streamed
        if on_text is None:
            return
        disp = [t for t in out_ids if t not in eos_ids]
        full = tokenizer.decode(disp)
        if len(full) > streamed:
            on_text(full[streamed:])
            streamed = len(full)

    _stream()
    while len(out_ids) < max_new_tokens and pending not in eos_ids:
        # ---- draft full-width block; feeding pending_ctx appends exactly the just-
        # committed positions to the draft KV cache (DFlash caches only ctx KV, never
        # block KV) -> correct absolute RoPE offsets, no trim needed.
        block = mx.array([[pending] + [mask_id] * (bs - 1)])
        head = drafter(block, pending_ctx, dcache, logits_start=1)[0][:cap]  # [cap, V] mask logits
        if temperature > 0.0:
            q_probs = mx.softmax(head / temperature, axis=-1)
            draft_arr = mx.random.categorical(head / temperature)
            mx.eval(draft_arr, q_probs)
        else:
            draft_arr = mx.argmax(head, axis=-1)
            mx.eval(draft_arr)
            q_probs = None
        drafted = [int(x) for x in draft_arr.tolist()]

        # ---- verify ----
        verify_ids = mx.array([[pending] + drafted])
        v_logits, v_fused = _run_target(target_model, verify_ids, cache, tap)
        mx.eval(v_logits, v_fused)
        target_forwards += 1

        if temperature > 0.0:
            n, repl = _spec_sample_accept(v_logits[0], drafted, q_probs, temperature)
            committed = drafted[:n] + [repl]             # accepted prefix + residual/bonus
        else:
            tt = [int(x) for x in mx.argmax(v_logits[0], axis=-1).tolist()]
            n = 0
            while n < len(drafted) and drafted[n] == tt[n]:
                n += 1
            committed = drafted[:n] + [tt[n]]            # accepted prefix + bonus
        accept_lengths.append(len(committed))

        # ---- update target cache + draft ctx ----
        trim = len(drafted) - n
        if trim > 0:
            for c in cache:
                if c is not None and hasattr(c, "trim"):
                    c.trim(trim)
        pending_ctx = v_fused[:, : n + 1, :]             # [anchor, accepted] -> next draft ctx

        for tok in committed:
            out_ids.append(tok)
            if tok in eos_ids:
                break
        pending = committed[-1]
        _stream()

    secs = time.time() - t0
    disp = [t for t in out_ids if t not in eos_ids]
    return GenResult(
        text=tokenizer.decode(disp),
        token_ids=out_ids,
        num_tokens=len(out_ids),
        num_rounds=len(accept_lengths),
        accept_lengths=accept_lengths,
        target_forwards=target_forwards,
        seconds=secs,
    )


def _make_target_cache(target):
    return target.make_cache()


def _run_target(target, ids: mx.array, cache, tap: list[int]):
    """ids: [1, L]. Returns (logits[1,L,V], fused_hidden[1,L,n_tap*H])."""
    return target.run(ids, cache, tap)


def _spec_sample_accept(v_logits, draft, q_probs, temperature):
    """Speculative-sampling acceptance (Leviathan/Chen 2023) for one verified block.

    ``v_logits`` [1+L, V] are the target logits at the verify positions; ``draft`` is the
    list of L sampled tokens; ``q_probs`` [>=L, V] the draft distributions they were sampled
    from. Each token is accepted w.p. ``min(1, p(x)/q(x))``; the first rejection stops the
    block and resamples from the residual ``norm(max(0, p-q))``; if all accept, a bonus is
    sampled from the target. Returns ``(n_accepted, replacement_token)``. This is the rule
    that makes the output an exact sample from the target's temperature-T distribution."""
    L = len(draft)
    p = mx.softmax(v_logits / temperature, axis=-1)            # [1+L, V]
    rows = mx.arange(L)
    idx = mx.array(draft)
    pd = p[rows, idx]                                          # target prob of each drafted token
    qd = q_probs[rows, idx]                                    # draft prob it was sampled from
    u = mx.random.uniform(shape=(L,))
    accepted = u < mx.minimum(1.0, pd / mx.maximum(qd, 1e-9))
    mx.eval(accepted)
    n = 0
    for a in accepted.tolist():
        if not a:
            break
        n += 1
    if n < L:
        resid = mx.maximum(p[n] - q_probs[n], 0.0)            # residual at the rejected position
        resid = resid / mx.maximum(resid.sum(), 1e-9)
        repl = int(mx.random.categorical(mx.log(resid + 1e-20)).item())
    else:
        repl = int(mx.random.categorical(v_logits[L] / temperature).item())  # target bonus
    return n, repl


def speculative_generate(
    target_model,
    tokenizer,
    drafter,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    confidence_threshold: float = 0.0,
    max_draft_tokens: int | None = 2,
    temperature: float = 0.0,
    seed: int | None = None,
    apply_chat_template: bool = True,
    on_text=None,
    verbose: bool = False,
) -> GenResult:
    """Speculative decoding (batch=1).

    ``temperature == 0`` → **greedy**: argmax draft, exact-argmax-match verify. Output is
    target-greedy by construction (up to fp tie-breaking on near-ties).

    ``temperature > 0`` → **speculative sampling** (the paper's setup, §2.1): each draft
    position is sampled from its temperature-scaled distribution q, then accepted with
    probability ``min(1, p(x)/q(x))`` against the target distribution p; on the first
    rejection the token is resampled from the residual ``norm(max(0, p-q))`` and the rest
    of the block is discarded; if all are accepted a bonus is sampled from the target. This
    preserves the target's temperature-T sampling distribution exactly (lossless), and
    accepts more per round than greedy (greedy's exact-match is the strictest possible rule).

    ``max_draft_tokens`` (``cap``) bounds how many of the 7-token block are drafted *and*
    verified per round: on Apple Silicon the verify cost grows with tokens and the marginal
    draft token rarely survives, so cap=2 is the measured optimum (the drafter only runs
    lm_head/markov over these ``cap`` positions). ``None`` = full block. ``confidence_threshold``
    > 0 truncates the block adaptively via the drafter's confidence head (cumulative survival).
    """
    if seed is not None:
        mx.random.seed(seed)
    cfg = drafter.config
    tap = list(cfg.target_layer_ids)
    k = cfg.block_size
    mask_id = cfg.mask_token_id
    cap = k if max_draft_tokens is None else max(1, min(max_draft_tokens, k))

    eos_ids = eos_token_ids(tokenizer)

    # --- tokenize prompt ---
    ids = encode_prompt(tokenizer, prompt, use_chat=apply_chat_template)
    prompt_ids = mx.array([ids])

    cache = _make_target_cache(target_model)
    ctx_caches = drafter.make_ctx_cache()
    t0 = time.time()

    # --- prefill ---
    logits, fused = _run_target(target_model, prompt_ids, cache, tap)
    n_cached = prompt_ids.shape[1]
    drafter.update_context(fused, ctx_offset=0, ctx_caches=ctx_caches)
    if temperature > 0.0:
        pending = int(mx.random.categorical(logits[0, -1] / temperature).item())
    else:
        pending = int(mx.argmax(logits[0, -1]).item())  # first committed token
    mx.eval([c.k for c in ctx_caches])

    out_ids: list[int] = [pending]
    accept_lengths: list[int] = []
    target_forwards = 1
    streamed = 0

    def _stream():
        nonlocal streamed
        if on_text is None:
            return
        disp = [t for t in out_ids if t not in eos_ids]
        full = tokenizer.decode(disp)
        if len(full) > streamed:
            on_text(full[streamed:])
            streamed = len(full)

    _stream()
    while len(out_ids) < max_new_tokens and pending not in eos_ids:
        # ---- 1. draft a block ----
        # The backbone runs full-width: block attention is bidirectional, so each
        # position's hidden depends on the whole block, and shrinking it would change
        # the distribution the drafter was trained on. But we only ever *verify* `cap`
        # tokens, so run the lm_head and the sequential markov head over just the first
        # `cap` positions instead of all `k` — the rest used to be computed and thrown
        # away every round (the dominant slice of drafter time at small caps).
        block_ids = [pending] + [mask_id] * (k - 1)
        noise = drafter.embed(mx.array([block_ids]))            # [1, k, H]
        block_hidden = drafter.backbone(noise, n_cached, ctx_caches)
        head_hidden = block_hidden[:, :cap, :]                  # only the verified positions
        base_logits = drafter.compute_logits(head_hidden)[0]    # [cap, V]
        if temperature > 0.0:
            draft_arr, q_probs = drafter.sample_block_probs(base_logits, pending, temperature)
            mx.eval(draft_arr, q_probs)
        else:
            draft_arr = drafter.sample_block(base_logits, first_prev_token=pending)
            mx.eval(draft_arr)
            q_probs = None
        draft = [int(x) for x in draft_arr.tolist()]

        # optional confidence-based truncation (adaptive block length, within cap).
        # Paper §3.2.1: c_k is the *conditional* survival prob of position k given the
        # prefix accepted; the prefix survival prob is the cumulative product
        # a_j = ∏_{i<=j} c_i (Eq 7-8). Keep extending the draft while a_j stays above
        # the threshold, i.e. while the next token is still likely to survive verify.
        if confidence_threshold > 0.0 and drafter.confidence_head is not None:
            prev_tokens = mx.array([pending] + draft[:-1])
            conf = mx.sigmoid(drafter.confidence_logits(head_hidden[0], prev_tokens))
            mx.eval(conf)
            surv, keep = 1.0, 0
            for i, c in enumerate(conf.tolist()):
                surv *= c
                if surv < confidence_threshold:
                    break
                keep = i + 1
            draft = draft[:keep]
        if not draft:
            draft = [int(draft_arr[0].item())]  # always propose >=1 (q_probs[0] still aligns)

        # ---- 2. verify with the target ----
        verify_ids = mx.array([[pending] + draft])             # [1, 1+len(draft)]
        v_logits, v_fused = _run_target(target_model, verify_ids, cache, tap)
        mx.eval(v_logits, v_fused)
        target_forwards += 1

        # ---- 3. accept ----
        if temperature > 0.0:
            n, repl = _spec_sample_accept(v_logits[0], draft, q_probs, temperature)
            committed = draft[:n] + [repl]                     # accepted prefix + residual/bonus
        else:
            tt = [int(x) for x in mx.argmax(v_logits[0], axis=-1).tolist()]
            n = 0
            while n < len(draft) and draft[n] == tt[n]:
                n += 1
            committed = draft[:n] + [tt[n]]                    # accepted prefix + bonus
        accept_lengths.append(len(committed))

        # ---- 4. update caches/context ----
        trim = len(draft) - n
        if trim > 0:
            for c in cache:
                if c is not None and hasattr(c, "trim"):
                    c.trim(trim)
        # commit [pending, accepted drafts] (positions n_cached..n_cached+n) as context
        drafter.update_context(
            v_fused[:, : n + 1, :], ctx_offset=n_cached, ctx_caches=ctx_caches
        )
        n_cached = n_cached + n + 1
        mx.eval([c.k for c in ctx_caches])

        for tok in committed:
            out_ids.append(tok)
            if tok in eos_ids:
                break
        pending = committed[-1]
        _stream()

        if verbose:
            print(f"  round {len(accept_lengths):3d}: drafted {len(draft)}, "
                  f"accepted {n}, committed {len(committed)}")

    secs = time.time() - t0
    # strip trailing eos for display
    disp = [t for t in out_ids if t not in eos_ids]
    text = tokenizer.decode(disp)
    return GenResult(
        text=text,
        token_ids=out_ids,
        num_tokens=len(out_ids),
        num_rounds=len(accept_lengths),
        accept_lengths=accept_lengths,
        target_forwards=target_forwards,
        seconds=secs,
    )
