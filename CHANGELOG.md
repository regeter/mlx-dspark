# Changelog

All notable changes to `mlx-dspark`. Versions follow [SemVer](https://semver.org/) (pre-1.0: minor-ish features land as patch bumps).

## [0.1.0] — serving & tooling

Turns mlx-dspark from a library + demo CLI into a usable local **tool** — serve a DSpark/DFlash
model to LM Studio, the `openai` SDK, or any OpenAI-compatible client. All additions keep the
lossless verify loop; the OpenAI surface is stdlib-only (no FastAPI/uvicorn added).

### Added
- **OpenAI-compatible API server** (`mlx_dspark.server`, `python -m mlx_dspark serve`). Point any
  OpenAI client / LM Studio / `openai` SDK at `http://host:port/v1`. Endpoints:
  `POST /v1/chat/completions` (streaming SSE **and** non-stream, **multi-turn**), `POST /v1/completions`,
  `GET /v1/models`, `GET /health`, `GET /metrics`. Serves `dspark` / `dflash` / `baseline` on one target.
  Params: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`, `stream`, optional `--api-key`,
  CORS. Spec-decode gain surfaced in an `x_mlx_dspark` block (accept length, tok/s) + `/metrics`.
  All generation runs on one dedicated thread (MLX arrays are thread/stream-affine).
- **Prefix caching** (in-memory + optional **SSD spill**) — reuse the shared conversation prefix's KV
  across turns instead of re-prefilling it. **~13× faster turn-2** on a ~750-token shared context
  (measured). On for `dspark`/`baseline` on dense (trimmable-KVCache) targets; falls back for DFlash
  and Gemma-4's rotating/sliding-window caches. Lossless to the same fp-tie standard as the rest of
  the project; invalidated on any error so it can't desync. Flags: `--no-prefix-cache`,
  `--prefix-cache-dir`, `--prefix-cache-max-ram-mb`.
- **Tool calling** — OpenAI `tools` / `tool_calls`, parsed from both native formats (Qwen3 Hermes-JSON
  and Gemma-4 `<|tool_call>call:…`), streamed as `delta.tool_calls`; inbound history normalized so
  prior tool calls render through the chat template.
- **Lossless top-p / top-k sampling** — nucleus/top-k truncation applied to both draft and target so
  temperature sampling stays an exact sample from the (truncated) target. Validated model-free.
- **Thinking toggle** — per-request `enable_thinking` / `chat_template_kwargs` and a server `--no-thinking`
  default (silences Qwen3 `<think>` blocks for a served endpoint).
- **Model-centric interface** — name the **target** with `--model <hf-repo | local-path>` (like
  mlx-lm); the matched drafter auto-resolves from a registry (quantization-agnostic), or pass
  `--drafter`. Replaces the old 2-value `--family`. `mlx-dspark models` lists targets with a known
  drafter. `--family` / `--target` / `load_pair("qwen3")` kept as **deprecated** aliases (still work).
- **Subcommand CLI** — `serve` / `generate` / `models` / `doctor` (env + model-fit check), plus a
  `mlx-dspark` console-script entry point. The old flat `python -m mlx_dspark --prompt …` still works.
- **Test suite** (`tests/`, 35 tests) covering the server protocol, streaming, stop sequences,
  tool-call parsing, top-p losslessness, and the prefix-cache manager — all model-free (fast, CI-friendly).

### API
- `generate()` functions gained `prompt_ids=`, `cache=`/`ctx_caches=`/`reuse_len=` (prefix reuse),
  `stop=`, `top_p=`, `top_k=`, and a `finish_reason` on `GenResult`; new `encode_messages()` (multi-turn).
  Backward compatible.

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
