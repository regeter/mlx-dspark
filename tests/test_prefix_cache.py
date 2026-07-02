"""Model-free tests for the prefix-cache manager: LCP reuse, trimming, the min-reuse gate,
bookkeeping (cache holds all-but-last generated token), reuse-eligibility detection, reset."""

from __future__ import annotations

from mlx_dspark.prefix_cache import PrefixCache, _lcp, target_cache_reusable


class KVCache:  # name matters: target_cache_reusable whitelists exactly "KVCache"
    def __init__(self, offset=0):
        self.offset = offset

    def trim(self, n):
        n = min(n, self.offset)
        self.offset -= n
        return n


class RotatingKVCache:  # a non-reusable type
    offset = 0

    def trim(self, n):
        return 0


class FakeCtx:
    def __init__(self):
        self.k = None
        self.v = None
        self.trimmed_to = None

    def trim_to(self, length):
        self.trimmed_to = length


def _mk_cache():
    return [KVCache(), KVCache()]


def _mk_ctx():
    return [FakeCtx(), FakeCtx()]


def test_lcp():
    assert _lcp([1, 2, 3], [1, 2, 9]) == 2
    assert _lcp([1, 2, 3], [1, 2, 3, 4]) == 3
    assert _lcp([], [1]) == 0
    assert _lcp([5], [6]) == 0


def test_reusable_detection():
    assert target_cache_reusable([KVCache(), KVCache()]) is True
    assert target_cache_reusable([KVCache(), RotatingKVCache()]) is False


def test_acquire_empty_is_fresh():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=4)
    cache, ctx, reuse_len = pc.acquire([1, 2, 3, 4, 5])
    assert reuse_len == 0 and len(cache) == 2 and len(ctx) == 2


def test_store_bookkeeping_and_reuse():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=4)
    prompt = list(range(1, 11))                 # 10 prompt tokens
    gen = [11, 12, 13]                           # 3 generated
    cache, ctx, _ = pc.acquire(prompt)
    for c in cache:                             # simulate post-generation cache length
        c.offset = len(prompt) + len(gen) - 1   # holds all but the last generated token = 12
    pc.store(cache, ctx, prompt, gen)
    assert pc.info()["cached_tokens"] == 12     # prompt + gen[:-1]

    # a follow-up prompt that diverges after 6 shared tokens -> reuse 6, trim caches to 6
    cache2, ctx2, reuse_len = pc.acquire([1, 2, 3, 4, 5, 6, 50, 51])
    assert reuse_len == 6
    assert cache2 is cache and all(c.offset == 6 for c in cache2)   # trimmed 12 -> 6
    assert all(c.trimmed_to == 6 for c in ctx2)
    assert pc.hits == 1 and pc.reused_tokens == 6


def test_min_reuse_gate():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=8)
    prompt = list(range(1, 11))
    cache, ctx, _ = pc.acquire(prompt)
    for c in cache:
        c.offset = len(prompt) - 1
    pc.store(cache, ctx, prompt, [99])
    # shares only 3 tokens (< min_reuse 8) -> fresh, no reuse
    _, _, reuse_len = pc.acquire([1, 2, 3, 500, 501, 502])
    assert reuse_len == 0 and pc.hits == 0


def test_reuse_len_capped_below_prompt_len():
    # even an identical prompt keeps >=1 token to prefill (need next-token logits)
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=1)
    prompt = [1, 2, 3, 4, 5]
    cache, ctx, _ = pc.acquire(prompt)
    for c in cache:
        c.offset = len(prompt) - 1
    pc.store(cache, ctx, prompt, [6])           # cached_tokens = [1,2,3,4,5]
    _, _, reuse_len = pc.acquire([1, 2, 3, 4, 5])
    assert reuse_len == 4                        # min(lcp=5, cached=5, len-1=4)


def test_reset_invalidates():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=1)
    cache, ctx, _ = pc.acquire([1, 2, 3])
    for c in cache:
        c.offset = 2
    pc.store(cache, ctx, [1, 2, 3], [4])
    pc.reset()
    assert pc.info()["cached_tokens"] == 0
    _, _, reuse_len = pc.acquire([1, 2, 3])
    assert reuse_len == 0
