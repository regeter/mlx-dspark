"""Unit tests for the model-free helpers in generate.py: the streaming/stop machinery,
finish-reason logic, and chat-template result normalization. No weights required.
"""

from __future__ import annotations

import mlx_dspark.generate as g


class _FakeTok:
    """decode(ids) == the string whose chars have those code points."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _run(stop, rounds, eos=frozenset()):
    tok = _FakeTok()
    chunks = []
    st = g._Streamer(tok, set(eos), chunks.append, stop)
    out = []
    for r in rounds:
        out += [ord(c) for c in r]
        st.update(out)
        if st.stopped:
            break
    st.flush()
    return "".join(chunks), st.text, st.stopped


def test_stream_no_stop_emits_everything():
    streamed, text, stopped = _run(None, ["Hel", "lo ", "world"])
    assert streamed == "Hello world" and text == "Hello world" and not stopped


def test_stop_within_round_cuts_and_no_leak():
    streamed, text, stopped = _run(["STOP"], ["abc", "deSTOPxyz"])
    assert text == "abcde" and stopped
    assert "STOP" not in streamed and streamed == "abcde"


def test_stop_straddling_rounds_held_back():
    # "ST" is emitted in round 1 but must be held back until we know it's part of "STOP"
    streamed, text, stopped = _run(["STOP"], ["abST", "OPcd"])
    assert text == "ab" and stopped and streamed == "ab"


def test_earliest_of_multiple_stops_wins():
    streamed, text, stopped = _run(["END", "STOP"], ["xxSTOPyyENDzz"])
    assert text == "xx" and stopped


class _StreamerLike:
    stopped = False


def test_finish_reason():
    s = _StreamerLike()
    assert g._finish_reason([1, 2, 3], 3, 9, {9}, s) == "stop"       # last token is eos
    stopped = _StreamerLike()
    stopped.stopped = True
    assert g._finish_reason([1, 2, 3], 100, 5, {9}, stopped) == "stop"  # a stop string hit
    assert g._finish_reason([1, 2, 3], 3, 5, {9}, _StreamerLike()) == "length"  # hit the cap
    assert g._finish_reason([1, 2], 3, 5, {9}, _StreamerLike()) == "stop"       # under cap, no eos


def test_topp_speculative_sampling_is_lossless():
    """The committed token must be an exact sample from top-p/top-k(softmax(target/T)),
    independent of the (deliberately mismatched) draft distribution q. This is the core
    losslessness guarantee for temperature + nucleus sampling."""
    import mlx.core as mx
    import numpy as np

    from mlx_dspark.generate import _spec_sample_accept
    from mlx_dspark.sampling import sample_probs, truncate_probs

    mx.random.seed(0)
    V = 6
    target = mx.array([2.0, 1.0, 0.5, 0.0, -1.0, -2.0])
    q = mx.softmax(mx.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]), axis=-1)  # ~reverse of target
    v_logits = mx.stack([target, target], axis=0)
    qrow = q.reshape(1, V)

    for (T, tp, tk) in [(1.0, 1.0, 0), (1.0, 0.8, 0), (0.7, 1.0, 3)]:
        exp = np.array(truncate_probs(mx.softmax(target / T, axis=-1), tp, tk).tolist())
        counts = np.zeros(V)
        for _ in range(20000):
            x = int(sample_probs(q).item())
            n, repl = _spec_sample_accept(v_logits, [x], qrow, T, tp, tk)
            counts[x if n == 1 else repl] += 1
        emp = counts / counts.sum()
        assert np.abs(emp - exp).max() < 0.02, (T, tp, tk, emp, exp)


def test_ids_from_template_result_shapes():
    assert g._ids_from_template_result([1, 2, 3]) == [1, 2, 3]
    assert g._ids_from_template_result([[4, 5, 6]]) == [4, 5, 6]

    class BatchEncoding(dict):
        pass

    be = BatchEncoding(input_ids=[[7, 8, 9]])
    assert g._ids_from_template_result(be) == [7, 8, 9]
