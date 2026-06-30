# Changelog

All notable changes to `mlx-dspark`. Versions follow [SemVer](https://semver.org/) (pre-1.0: minor-ish features land as patch bumps).

## [0.0.3]

### Added
- **z-lab DFlash drafter support** (block-diffusion speculative decoding). Run z-lab's original
  DFlash checkpoints natively on Apple Silicon through the same lossless verify loop as DSpark:
  - `load_dflash()`, `load_dflash_pair()`, `DFLASH_PRESETS`, `dflash_generate()`, and a
    `python -m mlx_dspark --mode dflash` CLI path (`--max-draft 0` = full block).
  - Presets: `gemma4` (`z-lab/gemma4-12B-it-DFlash`) and `qwen3` (`z-lab/Qwen3-4B-DFlash-b16`).
    Other z-lab adapters (e.g. `Qwen3-8B-DFlash-b16`) share the arch and load via `load_dflash(repo)`.
  - DFlash reuses the **target's** embed/lm-head (bound automatically); the drafter model classes
    are vendored from [z-lab/dflash](https://github.com/z-lab/dflash) (MIT) — see `NOTICE`.
  - Greedy **and** temperature>0 (lossless speculative sampling) for DFlash.
- **DSpark vs DFlash head-to-head** in the README (same target/Mac): DFlash's block-16 wins
  code/math (accept ~6, ~2.1×); DSpark's markov head wins open chat.

## [0.0.2]
### Added / changed
- Drafter-slice speedup (compute lm_head/markov over `cap` positions only) — output-neutral +9–10%.
- `--max-draft 2` is the new default (measured M-series optimum for both families).
- Lossless temperature speculative sampling (`--temperature`, paper §2.1).
- Optional 4-bit target (`--target ...-4bit`) for max absolute throughput / ≤24 GB Macs.

## [0.0.1]
- Initial release: DSpark speculative decoding for Apple Silicon (MLX), Gemma-4 12B + Qwen3-4B.
