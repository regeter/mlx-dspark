"""Model-free tests for prompt-lookup drafting: the n-gram index semantics."""

from __future__ import annotations

from mlx_dspark.lookup import NGramIndex


def _idx(tokens, **kw):
    ix = NGramIndex(**kw)
    ix.extend(list(tokens))
    return ix


def test_default_minimum_is_trigram():
    # bigram-only repetition must NOT fire at the defaults (bigrams are spurious on chat
    # text and rejected drafts cost wider forwards — measured net-negative on M-series)
    ix = _idx([5, 6, 7, 0, 1, 2, 5, 6])
    assert ix.propose() == []
    # a trigram match does fire
    ix = _idx([4, 5, 6, 7, 8, 0, 4, 5, 6], max_draft=2)
    assert ix.propose() == [7, 8]


def test_no_match_returns_empty():
    assert _idx([1, 2, 3, 4]).propose() == []


def test_simple_repeat_proposes_continuation():
    # ... 5 6 7 8 ... 5 6 -> propose [7, 8, ...] (bigram matching enabled explicitly)
    ix = _idx([5, 6, 7, 8, 9, 1, 5, 6], min_n=2, max_n=3, max_draft=3)
    assert ix.propose() == [7, 8, 9]


def test_self_occurrence_is_skipped():
    # the current suffix IS in the index (it was just inserted) — must not match itself
    ix = _idx([1, 2, 3, 4, 5], min_n=2, max_n=3)   # suffix (4,5) occurs only as the suffix
    assert ix.propose() == []


def test_latest_earlier_occurrence_wins():
    # (5,6) appears twice earlier with different continuations; the most recent wins
    ix = _idx([5, 6, 7, 0, 5, 6, 8, 0, 5, 6], min_n=2, max_n=3, max_draft=1)
    assert ix.propose() == [8]


def test_longer_ngram_preferred():
    # trigram (4,5,6) -> 9 is more specific than bigram (5,6) -> 7
    ix = _idx([4, 5, 6, 9, 0, 5, 6, 7, 0, 4, 5, 6], min_n=2, max_n=3, max_draft=1)
    assert ix.propose() == [9]


def test_max_draft_and_tail_truncation():
    ix = _idx([5, 6, 7, 8, 9, 10, 11, 0, 5, 6], min_n=2, max_n=3, max_draft=4)
    assert ix.propose() == [7, 8, 9, 10]
    # continuation runs into the end of the sequence -> shorter draft, never empty-pads
    ix2 = _idx([1, 9, 9, 2, 9, 9], min_n=2, max_n=3, max_draft=4)
    assert ix2.propose() == [2, 9, 9]


def test_incremental_extend_matches_bulk():
    toks = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 9]
    bulk = _idx(toks, min_n=2, max_n=3, max_draft=2)
    inc = NGramIndex(min_n=2, max_n=3, max_draft=2)
    for t in toks:
        inc.extend([t])
    assert bulk.propose() == inc.propose()
