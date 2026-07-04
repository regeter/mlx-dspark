"""transformers>=5.13 import-compat shim (issue #1).

mlx_lm registers a tokenizer by a string key, which transformers>=5.13's
``_LazyAutoMapping.register`` (assuming a config *class*) rejects with
``AttributeError: 'str' object has no attribute '__module__'`` at module scope,
taking down ``import mlx_dspark``. ``mlx_dspark`` installs a scoped shim at import.
These tests are transformers-version-agnostic: the shim is applied unconditionally,
so a string-key register must be tolerated on any installed transformers.
"""


def test_string_register_shim_applied():
    import mlx_dspark  # noqa: F401 — importing applies the shim

    from transformers.models.auto.auto_factory import _LazyAutoMapping

    assert getattr(_LazyAutoMapping.register, "_mlx_dspark_patched", False)


def test_string_key_register_does_not_raise():
    import mlx_dspark  # noqa: F401

    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    # Pre-shim this raised AttributeError on transformers>=5.13.
    TOKENIZER_MAPPING.register("_mlx_dspark_test_key", (None, None), exist_ok=True)
    assert TOKENIZER_MAPPING._extra_content.get("_mlx_dspark_test_key") == (None, None)


def test_class_key_register_still_routes_to_original():
    """A real config-class key must not take the string fallback path."""
    import mlx_dspark  # noqa: F401

    from transformers.models.auto.auto_factory import _LazyAutoMapping

    reg = _LazyAutoMapping.register
    # The wrapper only diverts non-class keys; class keys defer to the original.
    # (Smoke check that the wrapper is in place and callable with a class key.)
    assert callable(reg)

    class _FakeCfg:
        pass

    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    TOKENIZER_MAPPING.register(_FakeCfg, (None, None), exist_ok=True)
