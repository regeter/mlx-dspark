"""Model-free tests for OpenAI presence/frequency penalties (generate._Penalizer).

Validate the per-position [M+1, V] penalty a verify block subtracts: base completion counts
plus the block's own draft prefix, with presence firing once per token and frequency per count.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_dspark.generate import _Penalizer


def test_inactive_is_noop():
    p = _Penalizer(0.0, 0.0)
    assert not p.active
    logits = mx.random.normal((3, 8))
    assert p.apply(logits, [1, 2]) is logits            # identity object -> default path untouched


def test_base_penalty_presence_and_frequency():
    p = _Penalizer(presence=1.0, frequency=0.5)
    p.add([5, 5, 2])                                     # token 5 x2, token 2 x1
    pen = p.block_penalty(8, [], mx.float32)            # [1, 8]
    assert pen.shape == (1, 8)
    row = pen[0].tolist()
    assert abs(row[5] - (1.0 + 0.5 * 2)) < 1e-6         # presence + freq*count
    assert abs(row[2] - (1.0 + 0.5 * 1)) < 1e-6
    assert row[0] == 0.0 and row[3] == 0.0


def test_draft_prefix_new_token_gets_presence_then_frequency():
    p = _Penalizer(presence=1.0, frequency=0.5)          # no base counts
    pen = p.block_penalty(8, [3, 3], mx.float32)         # draft prefix [3, 3] -> [3, 8]
    assert pen.shape == (3, 8)
    col3 = [pen[i][3].item() for i in range(3)]
    assert col3[0] == 0.0                                # row0: base only, token 3 unseen
    assert abs(col3[1] - 1.5) < 1e-6                     # +presence+freq (first occurrence)
    assert abs(col3[2] - 2.0) < 1e-6                     # +freq (second occurrence)


def test_draft_prefix_token_already_in_base_only_frequency():
    p = _Penalizer(presence=1.0, frequency=0.5)
    p.add([7])                                           # base count 1
    pen = p.block_penalty(8, [7], mx.float32)            # [2, 8]
    col7 = [pen[i][7].item() for i in range(2)]
    assert abs(col7[0] - 1.5) < 1e-6                     # base: presence + freq*1
    assert abs(col7[1] - 2.0) < 1e-6                     # +freq only (already seen)


def test_apply_subtracts_penalty():
    p = _Penalizer(presence=2.0, frequency=0.0)
    p.add([1])
    logits = mx.zeros((2, 4))
    out = p.apply(logits, [3])                           # row0 penalizes token1; row1 also token3
    assert out[0].tolist() == [0.0, -2.0, 0.0, 0.0]
    assert out[1].tolist() == [0.0, -2.0, 0.0, -2.0]     # token3 newly seen in draft prefix
