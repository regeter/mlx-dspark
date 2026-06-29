"""Warm benchmark: greedy vs DSpark speculative (both warm), multi-trial.

  python benchmark.py            # gemma4
  python benchmark.py qwen3
"""
import sys, time, mlx.core as mx
from mlx_dspark.load import load_pair
from mlx_dspark.generate import (
    speculative_generate, _make_target_cache, eos_token_ids, encode_prompt,
)

family = sys.argv[1] if len(sys.argv) > 1 else "gemma4"
target, tok, drafter, cfg = load_pair(family)
eos = eos_token_ids(tok)
N = 100
PROMPTS = [
    "Explain how rainbows form.",
    "Write a Python function to check if a string is a palindrome.",
    "Give three tips for staying focused while working.",
]

def greedy(ids, n):
    cache = _make_target_cache(target)
    lg = target.plain(mx.array([ids]), cache); mx.eval(lg)
    nx = int(mx.argmax(lg[0, -1]).item()); out = [nx]; t = time.time()
    while len(out) < n and nx not in eos:
        lg = target.plain(mx.array([[nx]]), cache); mx.eval(lg)
        nx = int(mx.argmax(lg[0, -1]).item()); out.append(nx)
    return len(out), time.time() - t

print(f"[{family}] warming up (ramping clocks)...")
for _ in range(2):
    greedy(encode_prompt(tok, "Tell me about the sea.", True), 120)
    speculative_generate(target, tok, drafter, "Tell me about the sea.",
                         max_new_tokens=120, max_draft_tokens=3)

print(f"\n{'prompt':<6} {'greedy':>9} | {'cap':>3} {'spec':>9} {'accept':>7} {'speedup':>8}")
agg = {}
for i, p in enumerate(PROMPTS):
    ids = encode_prompt(tok, p, True)
    gt, gs = greedy(ids, N); gtps = gt / gs
    for cap in (2, 3, 4):
        res = speculative_generate(target, tok, drafter, p, max_new_tokens=N,
                                   max_draft_tokens=cap)
        sp = gs / res.seconds
        agg.setdefault(cap, []).append(sp)
        tag = f"P{i}" if cap == 2 else ""
        print(f"{tag:<6} {gtps:>7.1f}/s | {cap:>3} {res.tokens_per_sec:>7.1f}/s "
              f"{res.mean_accept_len:>7.2f} {sp:>7.2f}x")
print("\nmean speedup by cap:")
for cap, v in agg.items():
    print(f"  cap={cap}: {sum(v)/len(v):.2f}x")
