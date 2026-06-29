# mlx-dspark

**DSpark speculative decoding for Apple Silicon**, built on [MLX](https://github.com/ml-explore/mlx).

DSpark is DeepSeek's semi-autoregressive, EAGLE-family speculative-decoding drafter,
open-sourced in the [DeepSpec](https://github.com/deepseek-ai/DeepSpec) codebase and used to
accelerate DeepSeek-V4. This project ports the **inference path** to MLX so the published
drafter checkpoints run natively on a Mac.

**Supported families** (auto-detected from the drafter config):

| family | target | drafter | RAM |
|---|---|---|---|
| `gemma4` | `gemma-4-12B-it-8bit` | `deepseek-ai/dspark_gemma4_12b_block7` | ~32 GB+ |
| `qwen3`  | `Qwen3-4B-8bit`        | `deepseek-ai/dspark_qwen3_4b_block7`   | ~16 GB |

## How it works

- A **parallel backbone** (5 Gemma-4 layers) consumes the target model's hidden states
  (tapped at layers `[5,17,29,41,46]`, EAGLE3-style) and proposes a 7-token block at once.
- A **rank-256 Markov head** adds a previous-token correction, sampled sequentially — the only
  sequential cost, which kills "suffix decay" cheaply.
- A **confidence head** scores each draft position (optional adaptive block length).
- The target **verifies** every token, so output is **greedy-correct by construction**
  (identical to plain greedy decoding, up to floating-point tie-breaking on near-ties).

The drafter is loaded 1:1 from the HF checkpoint and **quantized to 4-bit** by default
(~1.8 GB) so it's cheap to run every round.

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## Use

```bash
# CLI — pick a family (downloads drafter + instruct target on first run)
python -m mlx_dspark --family qwen3  --prompt "Explain how rainbows form."
python -m mlx_dspark --family gemma4 --prompt "Explain how rainbows form." --max-new-tokens 256

# side-by-side demo: baseline (plain target) vs dspark (record each, stack)
python -m mlx_dspark --family qwen3 --mode baseline --prompt "..." --max-new-tokens 400
python -m mlx_dspark --family qwen3 --mode dspark   --prompt "..." --max-new-tokens 400
```

```python
from mlx_dspark import load_pair, speculative_generate

target, tok, drafter, cfg = load_pair("qwen3")   # or "gemma4"
res = speculative_generate(target, tok, drafter, "Explain how rainbows form.")
print(res.text, res.mean_accept_len, res.tokens_per_sec)
```

## Results (M4 Pro, 48 GB; 8-bit instruct target, 4-bit drafter; warm — `python benchmark.py`)

| family | drafter `d_0` | accept len | greedy (baseline) | dspark (this project) | speedup |
|---|---|---|---|---|---|
| **Gemma-4 12B** | ~82% | 2.5–3.6 | ~17 tok/s | ~28 tok/s | **~1.5–1.6×** |
| **Qwen3-4B**    | ~85% | 2.1–2.8 | ~49 tok/s | ~66 tok/s | **~1.3–1.4×** |

"greedy" = the plain target model decoding one token per forward (no drafter); "dspark" =
speculative decoding with the DSpark drafter. Both produce **identical** output — DSpark is just
faster (it diverges from sequential greedy only at logit-margin≈0 ties). Smaller/faster targets (Qwen3-4B) have a lower
per-token verify cost, so the optimal `--max-draft` is smaller (~2).

### Target choice matters

The drafter is trained against a specific **instruct** model in **bf16** — use the matching
instruct target (`gemma-4-12B-it` / `Qwen3-4B`), not the base model (base gave `d_0` ~47% vs
82%). Higher precision raises acceptance; 8-bit is the sweet spot. `-bf16` maximizes acceptance;
4-bit verifies faster.

### Tuning

On Apple Silicon the target verify cost grows with the number of tokens verified, so the
optimum is to verify ~= the acceptance length: `--max-draft 4` (default). `--max-draft <block>`
(full 7) is faithful but *slower* on M-series. `--confidence-threshold 0.6` instead truncates
the block adaptively via the drafter's confidence head.

## License

MIT — see [`LICENSE`](LICENSE). This is an independent MLX port of the inference path of
DeepSeek's DSpark drafter; see [`NOTICE`](NOTICE) for attribution. No model weights are bundled.
