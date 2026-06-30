"""mlx-dspark: DSpark speculative decoding for Apple Silicon (MLX).

DSpark (from DeepSeek's DeepSpec codebase) is a semi-autoregressive,
EAGLE-family speculative-decoding drafter:

  - a *parallel backbone* proposes base logits for all K draft positions at once,
  - a tiny *sequential head* (low-rank, previous-token-conditioned) corrects
    suffix decay,
  - a *confidence head* scores how likely each drafted token survives
    verification (used here for adaptive draft length instead of the
    server-side load-aware scheduler, which is irrelevant single-user).

This package targets single-user local inference on Apple Silicon.
"""

__version__ = "0.0.3"

from .config import DSparkConfig
from .dflash_model import DFlashConfig, DFlashDraftModel
from .load import (
    DEFAULT_DRAFTER,
    DEFAULT_TARGET,
    DFLASH_PRESETS,
    PRESETS,
    load_dflash,
    load_dflash_pair,
    load_drafter,
    load_pair,
    load_target,
)
from .generate import GenResult, dflash_generate, greedy_generate, speculative_generate
from .target import Target

__all__ = [
    "DSparkConfig",
    "DFlashConfig",
    "DFlashDraftModel",
    "Target",
    "load_drafter",
    "load_dflash",
    "load_target",
    "load_pair",
    "load_dflash_pair",
    "speculative_generate",
    "dflash_generate",
    "greedy_generate",
    "GenResult",
    "PRESETS",
    "DFLASH_PRESETS",
    "DEFAULT_TARGET",
    "DEFAULT_DRAFTER",
]
