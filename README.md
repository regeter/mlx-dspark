<p align="center">
  <img src="https://raw.githubusercontent.com/ARahim3/mlx-dspark/main/mlx-dspark.png" alt="mlx-dspark" width="440">
</p>

<p align="center">
  <b>DeepSeek's DSpark <i>and</i> z-lab's DFlash speculative decoding — running natively on Apple Silicon via <a href="https://github.com/ml-explore/mlx">MLX</a>.</b>
  <br>Lossless drafters — same output, just faster — for Gemma-4 12B and Qwen3 (4B/8B), plus any matched
  <br>z-lab DFlash adapter. Built-in DSpark-vs-DFlash head-to-head: the winner flips with the model —
  <br>DFlash's block-16 hits <b>~2.1×</b> on code/math on the big target, DSpark (<b>~1.4–1.6×</b>) wins chat and smaller ones.
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

mlx-dspark runs two EAGLE-family speculative-decoding drafters natively on Apple Silicon: DeepSeek's
**DSpark** (semi-autoregressive, from the [DeepSpec](https://github.com/deepseek-ai/DeepSpec) codebase,
used to accelerate DeepSeek-V4) and z-lab's **DFlash** (block diffusion). Both are **lossless** — the
target verifies every token, so output is identical to normal decoding — and run under one verify loop,
so you can benchmark them head-to-head. `pip install mlx-dspark` (full setup in [Install](#install)).

> **What this is *not*:** DeepSeek-V4 inference. The targets are dense models (Gemma-4, Qwen3) that DeepSeek
> published DSpark drafters for — so this runs their real drafter method on a Mac, but the model producing
> tokens is Gemma / Qwen, not V4. V4 Flash/Pro (MoE, batched serving) is DSpark's own headline use case and
> needs a V4 engine like [ds4](https://github.com/antirez/ds4).

**Built-in presets** (`--family`) — pick a drafter with `--mode dspark` (default) or `--mode dflash`:

| family | target (instruct, 8-bit) | DSpark drafter (`--mode dspark`) | DFlash drafter (`--mode dflash`) | peak RAM |
|---|---|---|---|---|
| `gemma4` | `gemma-4-12B-it-8bit` | `deepseek-ai/dspark_gemma4_12b_block7` | `z-lab/gemma4-12B-it-DFlash` | ~15 GB |
| `qwen3`  | `Qwen3-4B-8bit`        | `deepseek-ai/dspark_qwen3_4b_block7`   | `z-lab/Qwen3-4B-DFlash-b16`  | ~8 GB |

*Peak RAM* is measured on an M4 Pro (8-bit target + 4-bit drafter + KV cache). Add headroom for macOS:
gemma4 is comfortable on a **24 GB** Mac, qwen3 on **16 GB**. A 4-bit target (`--target …-it-4bit`) roughly
halves the target's share (gemma4 → ~9 GB, fits 16 GB). The DFlash drafter is even smaller than DSpark's,
so `--mode dflash` uses slightly less.

These are just the *convenience pairings*, not the limit. The DFlash path is a recipe: **any** z-lab
DFlash checkpoint with a matched dense Qwen3 / Gemma-4 target runs by pointing `--drafter` / `--target`
at the repos — e.g. `Qwen3-8B-DFlash-b16`, no code change (see
[Run any z-lab DFlash adapter](#run-any-z-lab-dflash-adapter)). The two drafters have different
trade-offs (DSpark's Markov head wins open chat; DFlash's block-16 wins code/math) — see
[DSpark vs DFlash](#dspark-vs-dflash--eagle3) for the head-to-head.

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

(That's DSpark. **DFlash** — `--mode dflash` — is a different drafter: a *block-diffusion* model that
denoises a whole 16-token block in one parallel pass and reuses the target's own embed/lm-head. Same
lossless verify loop, different trade-offs — see [DSpark vs DFlash](#dspark-vs-dflash--eagle3).)

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

# z-lab DFlash drafter instead (--max-draft 0 = full 16-block; best on code/math)
python -m mlx_dspark --mode dflash --family gemma4 --max-draft 0 --prompt "Write a binary search."

# sampled (not greedy) — lossless wrt the target at temperature T (works for dspark and dflash)
python -m mlx_dspark --family qwen3 --prompt "Write a short poem." --temperature 1.0 --seed 0
```

```python
from mlx_dspark import load_pair, speculative_generate

target, tok, drafter, cfg = load_pair("qwen3")   # or "gemma4"
res = speculative_generate(target, tok, drafter, "Explain how rainbows form.")
print(res.text, res.mean_accept_len, res.tokens_per_sec)
```

Run z-lab's **DFlash** drafter instead (block-diffusion; strongest on code/math at the full block):

```python
from mlx_dspark import load_dflash_pair, dflash_generate

target, tok, drafter, cfg = load_dflash_pair("gemma4")          # drafter bound to the target's head
res = dflash_generate(target, tok, drafter, "Write a binary search in Python.")  # max_draft_tokens=None = full 16-block
print(res.text, res.mean_accept_len, res.tokens_per_sec)
```

## Results (M4 Pro, warm; 8-bit instruct target, 4-bit drafter)

**DSpark** vs the **official MLX tools** running the same model (`mlx_lm.generate` / `mlx_vlm.generate`),
at its `cap=2` optimum:

| family | drafter `d_0` | accept len | baseline (official) | mlx-dspark | speedup |
|---|---|---|---|---|---|
| **Gemma-4 12B** | ~82% | ~2.5 | 18.4 tok/s | ~30 tok/s | **~1.6×** (≤2× on code/math) |
| **Qwen3-8B**    | –    | ~2.44 | 29.4 tok/s | ~47 tok/s | **~1.6×** |
| **Qwen3-4B**    | ~85% | ~2.25 | 52.9 tok/s | ~73 tok/s | **~1.4×** |

**DFlash** (z-lab) trades the other way — its block-16 denoise shines on **structured** content. On
Gemma-4 12B, full-block DFlash reaches **~2.1×** on code/math (accepted length ~6.0), while DSpark's
Markov head keeps the lead on **open chat** (1.65×). They're complementary:

| Gemma-4 12B (vs greedy ≈17.3 tok/s) | chat | code | math |
|---|---|---|---|
| DSpark (cap 2) | **1.65×** (acc 2.45) | 1.89× (2.78) | 1.89× (2.86) |
| DFlash (full 16) | 0.98× (2.68) | **2.10×** (5.95) | **2.12×** (6.20) |

**Which to use — it's the target's verify cost, not just the content:**

| target (verify cost) | DSpark (cap 2) | DFlash (full-16) | pick |
|---|---|---|---|
| Gemma-4 12B — *expensive verify* | 1.65× chat, ~1.9× code/math | **~2.1×** code/math, ~1.0× chat | DFlash on code/math, DSpark on chat |
| Qwen3-8B — *cheap verify* | **~1.6× everywhere** | ~0.9–1.1× (wash) | **DSpark** |

Bigger / slower-verify target → DFlash's full block pays off on code/math; smaller / fast target → DSpark
wins outright (confirmed against z-lab's own dflash-mlx runner — see [DSpark vs DFlash](#dspark-vs-dflash--eagle3)).

Full per-method table + analysis in [DSpark vs DFlash](#dspark-vs-dflash--eagle3).

All paths produce **identical** output to plain decoding — they're just faster (divergence from
sequential greedy happens only at logit-margin≈0 ties). (`python benchmark.py`'s in-harness greedy
baseline is ~5% slower than the official tools, so DSpark shows a slightly higher ~1.73× / ~1.45× —
we quote the conservative number.)

### What to expect on Apple Silicon (the speedup ceiling)

**These numbers are in line with the DSpark paper.** The paper's headline is **60–85% (V4-Flash)
/ 57–78% (V4-Pro) per-user speedup = ~1.57–1.85×**, measured in *batched production serving vs an
MTP-1 baseline* (where the confidence scheduler's job is avoiding batch-capacity waste). Our
~1.4–1.6× *single-user vs the official tools* sits in/near that band — Gemma-4 12B and Qwen3-8B land
inside it; the smaller Qwen3-4B is a touch below because its cheaper verify leaves less to amortize. The "2–4×"
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

**DSpark** (`--mode dspark`): the target verify cost grows per token *and* the marginal draft token
rarely survives, so the measured optimum for both families is **`--max-draft 2` (default)** — higher
caps verify more tokens for little extra acceptance and are slower. The drafter only runs its lm_head
+ Markov head over these `cap` positions (the backbone stays full-width for faithful bidirectional
block attention). `--max-draft <block>` (full 7) is faithful but *slower*. `--confidence-threshold 0.6`
instead truncates the block adaptively via the drafter's confidence head.

**DFlash** (`--mode dflash`): the trade-off flips with content. Use **`--max-draft 0`** (full 16-block,
its native design point) on **code/math**, where acceptance reaches ~6 and the long block stays fast;
use a short cap (`--max-draft 2`) on **open chat**, where the block doesn't fill and the full block
becomes a net loss. Both `--temperature` (lossless sampling) and greedy are supported, same as DSpark.

## DSpark vs DFlash / EAGLE3

These are **three drafters from the same DeepSpec codebase**, all EAGLE-family (a tiny drafter that
consumes the *target's hidden states*, not a standalone draft LLM). They differ only in how the
block is drafted:

- **EAGLE3** — autoregressive (token-by-token): high quality but draft latency grows with block size.
- **DFlash** — parallel (whole block in one pass): fast, but **suffix decay** (later positions are
  predicted independently and collide).
- **DSpark** — semi-autoregressive: the **DFlash backbone + a cheap rank-256 Markov head** that
  injects token-to-token dependency, fixing suffix decay at ~0.6 ms/round. A strict upgrade of DFlash.

Per the paper (accepted length, full block, temp=1.0), DSpark beats the DeepSpec DFlash by **+16–18%**
and EAGLE3 by **+27–31%**. DFlash and EAGLE3 are already in `mlx-vlm` for Gemma-4; **this is the first
MLX port of DSpark** — and it also runs **z-lab's original DFlash** drafters (see below).

### Run z-lab's original DFlash too (`--mode dflash`)

The DSpark loader already runs DeepSpec's `dflash_*_block7` as-is (same `*DSparkModel` arch with
`markov_rank=0`). But [z-lab](https://github.com/z-lab/dflash)'s **original DFlash** (Chen et al.,
[arXiv:2602.06036](https://arxiv.org/abs/2602.06036), MIT) is a different design: **block diffusion** —
a Qwen3-style backbone that **reuses the target's embed/lm-head** and denoises a whole **block of 16**
mask tokens in one parallel pass. mlx-dspark now loads those published checkpoints natively and runs
them through the same lossless verify loop, so you can benchmark the two head-to-head on one target/Mac:

```bash
python -m mlx_dspark --mode dflash --max-draft 0 --prompt "..."                  # gemma4, full 16-block (its sweet spot)
python -m mlx_dspark --mode dflash --family qwen3 --max-draft 0 --prompt "..."   # Qwen3-4B DFlash
python -m mlx_dspark --mode dflash --max-draft 0 --temperature 1.0 --prompt "..."  # lossless sampling
```

Built-in presets: `gemma4` (`z-lab/gemma4-12B-it-DFlash`) and `qwen3` (`z-lab/Qwen3-4B-DFlash-b16`).
Greedy by default; `--temperature > 0` switches to lossless speculative sampling (exact sample from
target@T). For any *other* z-lab adapter, see [Run any z-lab DFlash adapter](#run-any-z-lab-dflash-adapter).

**Head-to-head, measured on-device** (gemma-4-12B-it-8bit, M4 Pro, warm, greedy/lossless, 4 prompts/domain
— accepted length / tok·s; greedy baseline ≈ 17.3 tok/s):

| method | chat | code | math |
|---|---|---|---|
| **DSpark** (cap 2) | **2.45 / 28.5** | 2.78 / 32.8 | 2.86 / 32.4 |
| z-lab **DFlash** (cap 2) | 2.15 / 24.2 | 2.76 / 31.3 | 2.71 / 29.6 |
| z-lab **DFlash** (full 16) | 2.68 / 16.9 | **5.95 / 36.6** | **6.20 / 36.3** |

The two are **complementary**, and it lines up with the paper's framing:

- **DFlash's block-16 wins structured content on both axes** — accepted length ~**6.0** on code/math
  (DSpark's block-7 tops out ~2.8) *and* ~**2.1×** throughput, because high acceptance amortizes the
  16-wide verify.
- **DSpark wins open chat** — its rank-256 Markov head fixes suffix decay, so chat accepts 2.45 at
  **1.65×**; DFlash's block never fills on unpredictable text (full-16 chat is ~0.98×, a slight *loss*).
- On a single-user Mac the long block only converts to wall-clock when acceptance is high (verify cost
  grows per token here) — DFlash's design target is the cheap-verify **batched-serving** regime.

DFlash is greedy-lossless to the same standard as DSpark (it diverges from sequential greedy only at the
same fp-margin≈0 ties).

**But the winner is model-dependent, not just content-dependent** — on a *smaller, cheap-verify* target
the picture flips. Qwen3-8B-8bit (M4 Pro, warm, greedy, 3 prompts/domain, accept / tok·s; greedy ≈ 28.8):

| method | chat | code | math |
|---|---|---|---|
| **DSpark** (cap 2) | **2.38 / 45.7** | **2.55 / 48.8** | **2.40 / 46.1** |
| DFlash (cap 2) | 1.99 / 33.8 | 2.22 / 37.0 | 2.11 / 35.7 |
| DFlash (full 16) | 2.19 / 21.1 | 2.94 / 27.6 | 2.66 / 25.5 |

Here **DSpark wins everywhere (~1.6×)** while DFlash's block advantage largely evaporates: the full-16 block
is a net *loss* (~0.9×) and even its best short-block mode only reaches ~1.1–1.3× on code. Qwen3-8B's verify
is cheap, so the wide block costs more than it returns and acceptance never climbs (~2.9 on code vs 5.95 on
the 12B). DFlash's block-16 edge needs an *expensive*-verify target (a big model like the 12B, or batched GPU
serving); rule of thumb: **bigger / slower-verify target → DFlash full-block on code/math; smaller / fast → DSpark.**

Cross-checked against z-lab's own optimized runner [`dflash-mlx`](https://github.com/bstnxbt/dflash-mlx) on the
*identical* target + drafter: its baseline matches ours (29.3 tok/s), and its DFlash is also a net loss at the
full block (0.92× on code) and only ~1.08× in adaptive mode — *even with its hand-written Metal verify kernels*.
So this is DFlash at this model scale on Apple Silicon, not an artifact of mlx-dspark's verify loop.

Reproduce these (Qwen3-8B has no preset — point `--drafter`/`--target` at the repos; downloads on first run):

```bash
python -m mlx_dspark --mode dspark --drafter deepseek-ai/dspark_qwen3_8b_block7 \
  --target mlx-community/Qwen3-8B-8bit --prompt "Write a binary search in Python."
python -m mlx_dspark --mode dflash --max-draft 0 --drafter z-lab/Qwen3-8B-DFlash-b16 \
  --target mlx-community/Qwen3-8B-8bit --prompt "Write a binary search in Python."
```

### Run any z-lab DFlash adapter

The `gemma4` / `qwen3` presets are just convenience pairings — the DFlash path is a **recipe**, not a
fixed list. Every z-lab DFlash checkpoint shares the same architecture, so any of them runs by pointing
at the drafter repo + its matched instruct target (z-lab names them to match, e.g. `Qwen3-8B-DFlash-b16`
↔ `Qwen3-8B`). No preset or code change needed.

**From the CLI** — override `--drafter` / `--target` (the `--family` flag is then ignored):

```bash
python -m mlx_dspark --mode dflash --max-draft 0 \
  --drafter z-lab/Qwen3-8B-DFlash-b16 --target mlx-community/Qwen3-8B-8bit \
  --prompt "Explain quicksort."
```

**From Python:**

```python
from mlx_dspark import load_target, load_dflash, dflash_generate

target, tok  = load_target("mlx-community/Qwen3-8B-8bit")    # the matched instruct target
drafter, cfg = load_dflash("z-lab/Qwen3-8B-DFlash-b16")      # any z-lab DFlash repo (downloads on first use)
drafter.bind(target.model)                                   # DFlash reuses the target's embed + lm-head
res = dflash_generate(target, tok, drafter, "Explain quicksort.", max_draft_tokens=None)  # None = full block
print(res.text, res.mean_accept_len, res.tokens_per_sec)
```

The only requirement is a target whose hidden size matches the drafter's (the drafter has no embed/lm-head
of its own — it reuses the target's). **Scope:** measured on dense **Gemma-4 12B** and **Qwen3 4B / 8B**
targets — all lossless, no code change (Qwen3-8B is untied embeddings and loads via `bind` exactly the
same). Larger Gemma-4 variants share the identical code path (just a bigger download). z-lab also ships
MoE / linear-attention variants (`Qwen3.5-*`, `gpt-oss-*`, …); those targets route differently and use a
gated-delta KV rollback this port doesn't wire yet, so they need a bit more work — PRs welcome.

## License

MIT — see [`LICENSE`](LICENSE). This is an independent MLX port of the inference path of
DeepSeek's DSpark drafter; see [`NOTICE`](NOTICE) for attribution. No model weights are bundled.
