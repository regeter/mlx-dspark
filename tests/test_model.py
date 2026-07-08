"""Model-free unit tests for the DSpark drafter building blocks."""

from __future__ import annotations

import mlx.core as mx

from mlx_dspark.model import CtxCache, DSparkAttention


class _AttnCfg:
    """Minimal config for a GQA DSparkAttention (n_rep = 8/2 = 4)."""
    num_attention_heads = 8
    n_kv_heads = 2
    attn_head_dim = 16
    attention_k_eq_v = False
    use_v_norm = False
    scaling = 16 ** -0.5
    hidden_size = 32
    attention_bias = False
    rms_norm_eps = 1e-6
    rope_theta = 1e6
    rope_parameters = None


def test_attend_uses_native_gqa_equivalent_to_tiling():
    """The drafter attention relies on SDPA's native GQA/MQA broadcast (no `_repeat_kv`): tiling
    the K/V up to full heads over the whole context every round was O(n_rep · ctx) of wasted
    bandwidth that collapsed long-context drafting. Guard that the shipped n_kv-head path is
    numerically identical to explicitly tiling — so a refactor can't silently break GQA or bring
    the tiling waste back without this test noticing."""
    mx.random.seed(0)
    attn = DSparkAttention(_AttnCfg())
    mx.eval(attn.parameters())

    cache = CtxCache()
    attn.update_ctx(mx.random.normal((1, 5, _AttnCfg.hidden_size)), 0, cache)   # 5 ctx positions
    hidden = mx.random.normal((1, 3, _AttnCfg.hidden_size))                      # 3-position block
    block_offset = cache.length

    shipped = attn.attend(hidden, block_offset, cache)                           # native GQA

    # reference: identical math but tile K/V to full heads before SDPA (the old `_repeat_kv`)
    B, q_len, _ = hidden.shape
    q = attn.q_proj(hidden).reshape(B, q_len, attn.n_heads, attn.head_dim)
    q = attn.rope(attn.q_norm(q).transpose(0, 2, 1, 3), offset=block_offset)
    k_blk, v_blk = attn._kv(hidden)
    k_blk = attn.rope(k_blk, offset=block_offset)
    k = mx.concatenate([cache.k, k_blk], axis=2)
    v = mx.concatenate([cache.v, v_blk], axis=2)
    n_rep = attn.n_heads // attn.n_kv_heads

    def tile(x):
        b, nkv, s, d = x.shape
        return mx.broadcast_to(mx.expand_dims(x, 2), (b, nkv, n_rep, s, d)).reshape(b, nkv * n_rep, s, d)

    ref = mx.fast.scaled_dot_product_attention(q, tile(k), tile(v), scale=attn.scale)
    ref = attn.o_proj(ref.transpose(0, 2, 1, 3).reshape(B, q_len, -1))

    assert shipped.shape == (1, 3, _AttnCfg.hidden_size)
    assert mx.allclose(shipped, ref, atol=1e-5).item()
