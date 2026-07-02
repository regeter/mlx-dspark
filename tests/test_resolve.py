"""Tests for the target->drafter resolver (model-centric CLI/library interface)."""

from __future__ import annotations

import pytest

from mlx_dspark.load import resolve


def test_full_repo_auto_resolves_drafter():
    assert resolve("mlx-community/Qwen3-8B-8bit", mode="dspark") == (
        "mlx-community/Qwen3-8B-8bit", "deepseek-ai/dspark_qwen3_8b_block7")
    assert resolve("mlx-community/Qwen3-8B-8bit", mode="dflash")[1] == "z-lab/Qwen3-8B-DFlash-b16"


def test_quantization_agnostic():
    # the drafter matches the model, not the quant
    for repo in ("mlx-community/Qwen3-8B-4bit", "some-org/Qwen3-8B-bf16", "x/Qwen3-8B-8bit"):
        assert resolve(repo, mode="dspark")[1] == "deepseek-ai/dspark_qwen3_8b_block7"


def test_gemma_naming_variants():
    assert resolve("mlx-community/gemma-4-12B-it-4bit", mode="dspark")[1] == \
        "deepseek-ai/dspark_gemma4_12b_block7"


def test_no_cross_match_between_sizes():
    assert resolve("mlx-community/Qwen3-4B-8bit", mode="dspark")[1] == \
        "deepseek-ai/dspark_qwen3_4b_block7"
    assert resolve("mlx-community/Qwen3-8B-8bit", mode="dspark")[1] == \
        "deepseek-ai/dspark_qwen3_8b_block7"


def test_legacy_family_alias():
    assert resolve("qwen3", mode="dspark")[0] == "mlx-community/Qwen3-4B-8bit"
    assert resolve(None, mode="dspark", family="gemma4")[0] == "mlx-community/gemma-4-12B-it-8bit"


def test_legacy_target_alias():
    assert resolve(None, mode="dflash", target="mlx-community/Qwen3-8B-8bit")[1] == \
        "z-lab/Qwen3-8B-DFlash-b16"


def test_explicit_drafter_override_any_target():
    assert resolve("my/Custom", mode="dspark", drafter="my/drafter") == ("my/Custom", "my/drafter")


def test_baseline_has_no_drafter():
    assert resolve("mlx-community/Qwen3-8B-8bit", mode="baseline") == (
        "mlx-community/Qwen3-8B-8bit", None)


def test_unknown_target_without_drafter_errors():
    with pytest.raises(ValueError) as e:
        resolve("my/Unknown-Model", mode="dspark")
    assert "no built-in" in str(e.value) and "--drafter" in str(e.value)


def test_local_path_basename_matched():
    # a local path is matched by its basename
    assert resolve("/models/Qwen3-8B-8bit", mode="dspark")[1] == "deepseek-ai/dspark_qwen3_8b_block7"
