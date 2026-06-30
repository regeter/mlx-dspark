<p align="center">
  <img src="https://raw.githubusercontent.com/ARahim3/mlx-dspark/main/mlx-dspark.png" alt="mlx-dspark" width="440">
</p>

<p align="center">
  <b>DeepSeek's DSpark speculative decoding — running natively on Apple Silicon via <a href="https://github.com/ml-explore/mlx">MLX</a>.</b>
  <br>A lossless drafter that makes Gemma-4 12B and Qwen3-4B faster on a Mac — <b>~1.6× / ~1.4×</b>, same output.
</p>

<p align="center">
  <a href="https://pypi.org/project/mlx-dspark/"><img src="https://img.shields.io/pypi/v/mlx-dspark?color=2563eb" alt="PyPI"></a>
  <img src="https://img.shields.io/pypi/pyversions/mlx-dspark" alt="Python">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-111111?logo=apple&logoColor=white" alt="Apple Silicon">
  <a href="https://github.com/ARahim3/mlx-dspark/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ARahim3/mlx-dspark/main/docs/demo.gif" alt="Baseline vs DSpark — same output, ~1.8x faster on Gemma-4 12B" width="840">
</p>

```bash
pip install mlx-dspark
```

DSpark is DeepSeek's semi-autoregressive, EAGLE-family speculative-decoding drafter, open-sourced in
the [DeepSpec](https://github.com/deepseek-ai/DeepSpec) codebase and used to accelerate DeepSeek-V4.
This ports the **inference path** to MLX so the published drafter checkpoints run natively on a Mac —
**losslessly** (the target verifies every token, so output is identical to normal decoding).

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
pip install mlx-dspark          # or:  uv pip install mlx-dspark
```

Apple Silicon + Python ≥ 3.10. Model weights download from the Hugging Face cache on first use
(none bundled).

From source (dev):

```bash
git clone https://github.com/ARahim3/mlx-dspark && cd mlx-dspark
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Use

```bash
# CLI — pick a family (downloads drafter + instruct target on first run)
python -m mlx_dspark --family qwen3  --prompt "Explain how rainbows form."
python -m mlx_dspark --family gemma4 --prompt "Explain how rainbows form." --max-new-tokens 256

# side-by-side demo: baseline (plain target) vs dspark (record each, stack)
python -m mlx_dspark --family qwen3 --mode baseline --prompt "..." --max-new-tokens 400
python -m mlx_dspark --family qwen3 --mode dspark   --prompt "..." --max-new-tokens 400

# sampled (not greedy) — lossless wrt the target at temperature T (paper's method)
python -m mlx_dspark --family qwen3 --prompt "Write a short poem." --temperature 1.0 --seed 0
```

```python
from mlx_dspark import load_pair, speculative_generate

target, tok, drafter, cfg = load_pair("qwen3")   # or "gemma4"
res = speculative_generate(target, tok, drafter, "Explain how rainbows form.")
print(res.text, res.mean_accept_len, res.tokens_per_sec)
```

## Results (M4 Pro, warm; 8-bit instruct target, 4-bit drafter, cap=2)

Speedup is vs the **official MLX tools** running the same model (`mlx_lm.generate` / `mlx_vlm.generate`):

| family | drafter `d_0` | accept len | baseline (official) | mlx-dspark | speedup |
|---|---|---|---|---|---|
| **Gemma-4 12B** | ~82% | ~2.5 | 18.4 tok/s | ~30 tok/s | **~1.6×** (≤2× on code/math) |
| **Qwen3-4B**    | ~85% | ~2.25 | 52.9 tok/s | ~73 tok/s | **~1.4×** |

Both produce **identical** output — DSpark is just faster (it diverges from sequential greedy only
at logit-margin≈0 ties). (`python benchmark.py`'s in-harness greedy baseline is ~5% slower than the
official tools, so it shows a slightly higher ~1.73× / ~1.45× — we quote the conservative number.)

### What to expect on Apple Silicon (the speedup ceiling)

**These numbers are in line with the DSpark paper.** The paper's headline is **60–85% (V4-Flash)
/ 57–78% (V4-Pro) per-user speedup = ~1.57–1.85×**, measured in *batched production serving vs an
MTP-1 baseline* (where the confidence scheduler's job is avoiding batch-capacity waste). Our
~1.4–1.6× *single-user vs the official tools* sits in/near that band — Gemma-4 12B lands inside it;
the smaller Qwen3-4B is a touch below because its cheaper verify leaves less to amortize. The "2–4×"
you may have seen elsewhere comes from other speculative-decoding papers on datacenter GPUs with
greedy baselines, not from DSpark's own claims.

Why it can't go much higher here: speculative decoding amortizes a *memory-bound* single-token
decode across the K tokens verified in one forward. On a datacenter GPU that arbitrage is huge
(parallel verify is nearly free, so speedup ≈ acceptance length). On an M-series chip it is much
weaker — verify cost **grows with the number of tokens** (measured ≈ +14 ms/token for Gemma-4 12B,
+1.5 ms/token for Qwen3-4B; multi-token verify drops out of the fast quantized GEMV path). With the
cost model `tok/s ≈ A / (drafter + 0.035 + slope·C)`, even a *perfect* drafter accepting the whole
7-token block tops out around **~2.2×** here. The binding limiter is acceptance length, set by the
drafter↔target match — **not** drafter quantization (4/8-bit/bf16 drafter all give identical
acceptance; 4-bit is simply fastest).

**Greedy vs sampling.** The default (`temperature 0`) is greedy: exact argmax-match verification,
output byte-identical to greedy decoding. `--temperature > 0` switches to the paper's actual method
(§2.1) — **speculative sampling**: draft tokens are sampled and accepted with prob `min(1, p/q)`,
rejections resample from the residual `norm(max(0, p−q))`, so the output is an **exact sample from
the target at temperature T** (lossless). It accepts a bit more per round (greedy's exact-match is
the strictest rule), but on M-series the verify cost still keeps cap≈2 optimal, where that extra
acceptance lives mostly in the unreached tail — so net speed is ≈ the greedy ratio. Use it when you
want *sampled* output, not for extra speed.

### Target choice matters

The drafter is trained against a specific **instruct** model — use the matching instruct target
(`gemma-4-12B-it` / `Qwen3-4B`), not the base model (base gave `d_0` ~47% vs 82%). Higher target
precision raises acceptance a little; **a bf16 target is *not* a speed win on M-series** — verify
dominates and roughly doubles, outpacing the small acceptance gain.

Target precision is a speed/quality knob. Since verify dominates, a **4-bit target** gives the
**highest absolute throughput** (and fits ≤24 GB Macs) — but the spec *ratio* shrinks (cheap verify
= less to amortize) and it's a lower-quality model:

| family | 8-bit target (default) | 4-bit target |
|---|---|---|
| gemma4 | greedy 17.5 → spec 30 tok/s (**1.73×**) | greedy 30.6 → spec 34–38 tok/s (1.1–1.25×) |
| qwen3  | greedy 49.8 → spec 73 tok/s (**1.45×**) | greedy 82 → spec 96–103 tok/s (1.17–1.26×) |

So: **8-bit for the biggest spec benefit + best quality; 4-bit for max absolute speed or small RAM**
(`--target mlx-community/gemma-4-12B-it-4bit`). Drafter stays 4-bit either way.

### Tuning

The target verify cost grows per token *and* the marginal draft token rarely survives, so the
measured optimum for both families is **`--max-draft 2` (default)** — higher caps verify more
tokens for little extra acceptance and are slower. The drafter only runs its lm_head + Markov head
over these `cap` positions (the backbone stays full-width for faithful bidirectional block
attention). `--max-draft <block>` (full 7) is faithful but *slower*. `--confidence-threshold 0.6`
instead truncates the block adaptively via the drafter's confidence head.

## DSpark vs DFlash / EAGLE3

These are **three drafters from the same DeepSpec codebase**, all EAGLE-family (a tiny drafter that
consumes the *target's hidden states*, not a standalone draft LLM). They differ only in how the
block is drafted:

- **EAGLE3** — autoregressive (token-by-token): high quality but draft latency grows with block size.
- **DFlash** — parallel (whole block in one pass): fast, but **suffix decay** (later positions are
  predicted independently and collide).
- **DSpark** — semi-autoregressive: the **DFlash backbone + a cheap rank-256 Markov head** that
  injects token-to-token dependency, fixing suffix decay at ~0.6 ms/round. A strict upgrade of DFlash.

Per the paper (accepted length, full block, temp=1.0), DSpark beats DFlash by **+16–18%** and EAGLE3
by **+27–31%**. DFlash and EAGLE3 are already in `mlx-vlm` for Gemma-4; **this is the first MLX port
of DSpark.**

Measured here (greedy, same Mac, DSpark vs DFlash — `dflash_*_block7` is the same architecture with
`markov_rank=0`, so this repo runs it as-is):

| | cap=2 (M-series optimum) | longer block (cap 3–4) |
|---|---|---|
| qwen3 chat | tied (~71 tok/s) | DSpark **+16–18%** accept |
| gemma4 chat | tied (~28 tok/s) | DSpark **+10–11%** accept |
| code / math | tied | tied (DFlash's parallel draft is already strong on predictable text) |

So the Markov head's advantage reproduces on-device — but mainly on **open-ended chat** and at
**longer blocks**. At the cap≈2 that Apple Silicon's verify cost forces, DSpark and DFlash run
effectively tied; DSpark is never worse (accept ≥ DFlash at negligible cost), it just needs the cheap
verify of GPU batched serving to separate. (The paper's larger gaps are temp=1.0 over full benchmark
suites; greedy single-prompt narrows them.)

## License

MIT — see [`LICENSE`](LICENSE). This is an independent MLX port of the inference path of
DeepSeek's DSpark drafter; see [`NOTICE`](NOTICE) for attribution. No model weights are bundled.
