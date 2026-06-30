"""CLI: run DSpark speculative decoding on Apple Silicon (streams tokens live).

Side-by-side demo (record each, then stack the two screen captures):

  # left panel — plain target, no drafter
  python -m mlx_dspark --mode baseline --prompt "Explain how rainbows form." --max-new-tokens 220

  # right panel — DSpark speculative decoding (same prompt, same output, faster)
  python -m mlx_dspark --mode dspark   --prompt "Explain how rainbows form." --max-new-tokens 220
"""

from __future__ import annotations

import argparse
import sys
import time

from .generate import dflash_generate, greedy_generate, speculative_generate
from .load import DFLASH_PRESETS, PRESETS, load_dflash, load_drafter, load_target


def _emit(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(prog="mlx_dspark")
    ap.add_argument("--mode", choices=["dspark", "dflash", "baseline"], default="dspark",
                    help="dspark = DSpark spec decoding; dflash = z-lab DFlash (block diffusion); "
                         "baseline = plain greedy target")
    ap.add_argument("--family", choices=["gemma4", "qwen3"], default="gemma4",
                    help="model preset (target + drafter); overridden by --target/--drafter")
    ap.add_argument("--prompt", default="Explain how rainbows form, in a few sentences.")
    ap.add_argument("--target", default=None)
    ap.add_argument("--drafter", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--max-draft", type=int, default=2,
                    help="tokens verified per round (cap). For --mode dflash, <=0 means the full "
                         "block (its native operating point — strongest on code/math).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (exact); >0 = speculative sampling (paper setup, lossless wrt target@T)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--drafter-bits", type=int, default=4)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args()
    _presets = DFLASH_PRESETS if args.mode == "dflash" else PRESETS
    target_repo = args.target or _presets[args.family]["target"]
    drafter_repo = args.drafter or _presets[args.family]["drafter"]

    labels = {"dspark": "DSpark speculative", "dflash": "DFlash (z-lab) speculative",
              "baseline": "Baseline (plain greedy)"}
    label = labels[args.mode]
    print(f"loading {args.mode}: target={target_repo}"
          + (f", drafter={drafter_repo}" if args.mode != "baseline" else ""))
    target, tok = load_target(target_repo)
    drafter = None
    if args.mode == "dspark":
        drafter, _ = load_drafter(drafter_repo, quantize=args.drafter_bits > 0,
                                  bits=max(args.drafter_bits, 2))
    elif args.mode == "dflash":
        drafter, _ = load_dflash(drafter_repo, quantize=args.drafter_bits > 0,
                                 bits=max(args.drafter_bits, 2))
        drafter.bind(target.model)

    on_text = None if args.no_stream else _emit
    print("\n" + "=" * 64)
    print(f"  ▶  {label}   ·   {target_repo.split('/')[-1]}")
    print("=" * 64)

    if args.mode == "dspark":
        res = speculative_generate(
            target, tok, drafter, args.prompt,
            max_new_tokens=args.max_new_tokens, max_draft_tokens=args.max_draft,
            confidence_threshold=args.confidence_threshold,
            temperature=args.temperature, seed=args.seed,
            apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = f" · accept {res.mean_accept_len:.2f}/round · {res.target_forwards} target fwds"
    elif args.mode == "dflash":
        res = dflash_generate(
            target, tok, drafter, args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_draft_tokens=(None if args.max_draft <= 0 else args.max_draft),
            temperature=args.temperature,
            seed=args.seed, apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = f" · accept {res.mean_accept_len:.2f}/round · {res.target_forwards} target fwds"
    else:
        res = greedy_generate(
            target, tok, args.prompt, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, seed=args.seed,
            apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = ""
    if args.no_stream:
        print(res.text)

    print("\n" + "-" * 64)
    print(f"  {res.num_tokens} tokens · {res.seconds:.2f}s · "
          f"\033[1m{res.tokens_per_sec:.1f} tok/s\033[0m{extra}")
    print("-" * 64)


if __name__ == "__main__":
    main()
