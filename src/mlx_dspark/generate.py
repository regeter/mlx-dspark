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


def greedy_generate(
    target_model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    apply_chat_template: bool = True,
    on_text=None,
) -> GenResult:
    """Plain greedy decoding of the target (no drafter, no hidden-state capture) —
    the fair 'run the model normally' baseline. Streams via on_text."""
    eos_ids = eos_token_ids(tokenizer)
    ids = encode_prompt(tokenizer, prompt, use_chat=apply_chat_template)
    cache = target_model.make_cache()

    t0 = time.time()
    logits = target_model.plain(mx.array([ids]), cache)
    nxt = int(mx.argmax(logits[0, -1]).item())
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
        nxt = int(mx.argmax(logits[0, -1]).item())
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


def _make_target_cache(target):
    return target.make_cache()


def _run_target(target, ids: mx.array, cache, tap: list[int]):
    """ids: [1, L]. Returns (logits[1,L,V], fused_hidden[1,L,n_tap*H])."""
    return target.run(ids, cache, tap)


def speculative_generate(
    target_model,
    tokenizer,
    drafter,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    confidence_threshold: float = 0.0,
    max_draft_tokens: int | None = 4,
    apply_chat_template: bool = True,
    on_text=None,
    verbose: bool = False,
) -> GenResult:
    """Greedy speculative decoding. Output is target-greedy by construction (up to
    fp tie-breaking on near-ties). ``max_draft_tokens`` caps how many of the 7-token
    block are verified per round; on Apple Silicon the target verify cost grows with
    tokens, so the optimum is ~= acceptance length (default 4). ``None`` = full block
    (faithful but slower on M-series). ``confidence_threshold`` > 0 instead truncates
    the block adaptively using the drafter's confidence head."""
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
        block_ids = [pending] + [mask_id] * (k - 1)
        noise = drafter.embed(mx.array([block_ids]))            # [1, k, H]
        block_hidden = drafter.backbone(noise, n_cached, ctx_caches)
        base_logits = drafter.compute_logits(block_hidden)[0]   # [k, V]
        draft = drafter.sample_block(base_logits, first_prev_token=pending)
        mx.eval(draft)
        draft = [int(x) for x in draft.tolist()]

        # optional confidence-based truncation (adaptive block length)
        if confidence_threshold > 0.0 and drafter.confidence_head is not None:
            prev_tokens = mx.array([pending] + draft[:-1])
            conf = mx.sigmoid(drafter.confidence_logits(block_hidden[0], prev_tokens))
            mx.eval(conf)
            below = [i for i, c in enumerate(conf.tolist()) if c < confidence_threshold]
            if below:
                draft = draft[: below[0]]
        if cap < len(draft):
            draft = draft[:cap]
        if not draft:
            draft = [int(mx.argmax(base_logits[0]).item())]  # always propose >=1

        # ---- 2. verify with the target ----
        verify_ids = mx.array([[pending] + draft])             # [1, 1+len(draft)]
        v_logits, v_fused = _run_target(target_model, verify_ids, cache, tap)
        mx.eval(v_logits, v_fused)
        target_forwards += 1
        tt = mx.argmax(v_logits[0], axis=-1)                   # [1+len(draft)]
        tt = [int(x) for x in tt.tolist()]

        # ---- 3. accept matching prefix + bonus ----
        n = 0
        while n < len(draft) and draft[n] == tt[n]:
            n += 1
        bonus = tt[n]                                          # correction / continuation
        committed = draft[:n] + [bonus]
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
