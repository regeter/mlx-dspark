"""mlx-dspark command line.

Subcommands:
  serve      Start an OpenAI-compatible API server (LM Studio / openai SDK / curl).
  generate   One-shot local generation (DSpark / DFlash / baseline). This is also the
             default when no subcommand is given, so the historical flat invocation
             ``python -m mlx_dspark --prompt ...`` keeps working unchanged.
  models     List the built-in target+drafter presets.
  doctor     Check the environment (Apple Silicon, MLX stack, RAM vs. model size).

Run ``mlx-dspark <cmd> -h`` for a command's flags.
"""

from __future__ import annotations

import argparse
import sys
import time

from .load import REGISTRY, resolve

_SUBCOMMANDS = ("serve", "generate", "models", "doctor")


def _emit(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


# --------------------------------------------------------------------------- generate


def cmd_generate(argv: list[str]) -> None:
    from .generate import dflash_generate, greedy_generate, speculative_generate
    from .load import load_dflash, load_drafter, load_target

    ap = argparse.ArgumentParser(prog="mlx-dspark generate")
    ap.add_argument("--mode", choices=["dspark", "dflash", "baseline"], default="dspark",
                    help="dspark = DSpark spec decoding; dflash = z-lab DFlash (block diffusion); "
                         "baseline = plain greedy target")
    ap.add_argument("--model", default=None,
                    help="target model: an HF repo or local path (e.g. mlx-community/Qwen3-8B-8bit). "
                         "The matched drafter auto-resolves for known targets; else pass --drafter.")
    ap.add_argument("--drafter", default=None, help="drafter repo/path (overrides auto-resolve)")
    ap.add_argument("--family", choices=["gemma4", "qwen3"], default=None,
                    help=argparse.SUPPRESS)          # deprecated alias for --model
    ap.add_argument("--target", default=None, help=argparse.SUPPRESS)  # deprecated alias for --model
    ap.add_argument("--prompt", default="Explain how rainbows form, in a few sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--max-draft", type=int, default=2,
                    help="tokens verified per round (cap). For --mode dflash, <=0 means the full "
                         "block (its native operating point — strongest on code/math).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (exact); >0 = speculative sampling (paper setup, lossless wrt target@T)")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling (temperature > 0)")
    ap.add_argument("--top-k", type=int, default=0, help="top-k sampling (temperature > 0)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--drafter-bits", type=int, default=4)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args(argv)

    try:
        target_repo, drafter_repo = resolve(args.model, mode=args.mode, drafter=args.drafter,
                                            family=args.family, target=args.target)
    except ValueError as e:
        ap.error(str(e))

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
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed,
            apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = f" · accept {res.mean_accept_len:.2f}/round · {res.target_forwards} target fwds"
    elif args.mode == "dflash":
        res = dflash_generate(
            target, tok, drafter, args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_draft_tokens=(None if args.max_draft <= 0 else args.max_draft),
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            seed=args.seed, apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = f" · accept {res.mean_accept_len:.2f}/round · {res.target_forwards} target fwds"
    else:
        res = greedy_generate(
            target, tok, args.prompt, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed,
            apply_chat_template=not args.no_chat_template, on_text=on_text,
        )
        extra = ""
    if args.no_stream:
        print(res.text)

    print("\n" + "-" * 64)
    print(f"  {res.num_tokens} tokens · {res.seconds:.2f}s · "
          f"\033[1m{res.tokens_per_sec:.1f} tok/s\033[0m{extra}")
    print("-" * 64)


# --------------------------------------------------------------------------- serve


def cmd_serve(argv: list[str]) -> None:
    from .server import Engine, run_server

    ap = argparse.ArgumentParser(prog="mlx-dspark serve",
                                 description="OpenAI-compatible API server.")
    ap.add_argument("--mode", choices=["dspark", "dflash", "baseline"], default="dspark")
    ap.add_argument("--model", default=None,
                    help="target model: an HF repo or local path (e.g. mlx-community/Qwen3-8B-8bit). "
                         "Matched drafter auto-resolves for known targets; else pass --drafter. "
                         "See `mlx-dspark models`.")
    ap.add_argument("--drafter", default=None, help="drafter repo/path (overrides auto-resolve)")
    ap.add_argument("--family", choices=["gemma4", "qwen3"], default=None,
                    help=argparse.SUPPRESS)          # deprecated alias for --model
    ap.add_argument("--target", default=None, help=argparse.SUPPRESS)  # deprecated alias for --model
    ap.add_argument("--max-draft", type=int, default=None,
                    help="cap on tokens verified per round; <=0 = full block. Default: dspark=2, "
                         "dflash=full block.")
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--drafter-bits", type=int, default=4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=None,
                    help="if set, requests must send 'Authorization: Bearer <key>'")
    ap.add_argument("--no-thinking", action="store_true",
                    help="default responses to non-thinking mode (Qwen3 enable_thinking=False); "
                         "clients can still override per-request")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="disable multi-turn prefix caching (reuse the shared conversation "
                         "prefix's KV; on by default for dspark/baseline on dense targets)")
    ap.add_argument("--prefix-cache-dir", default=None,
                    help="directory for the L2 SSD spill tier (enables spilling the cache to disk)")
    ap.add_argument("--prefix-cache-max-ram-mb", type=int, default=0,
                    help="spill the prefix cache to --prefix-cache-dir once it exceeds this many MB "
                         "of RAM (0 = never spill; requires --prefix-cache-dir)")
    args = ap.parse_args(argv)

    md = args.max_draft
    if md is None:
        max_draft = None                                   # engine picks the mode default
    elif args.mode == "dflash" and md <= 0:
        max_draft = None                                   # full block
    else:
        max_draft = max(1, md)

    print(f"loading {args.mode} engine — first run downloads weights…")
    try:
        engine = Engine.load(
            mode=args.mode, model=args.model, drafter=args.drafter,
            family=args.family, target=args.target,
            drafter_bits=args.drafter_bits, max_draft_tokens=max_draft,
            confidence_threshold=args.confidence_threshold,
            enable_thinking=False if args.no_thinking else None,
            prefix_cache=not args.no_prefix_cache,
            prefix_cache_dir=args.prefix_cache_dir,
            prefix_cache_max_ram_mb=args.prefix_cache_max_ram_mb,
        )
    except ValueError as e:
        ap.error(str(e))
    run_server(engine, host=args.host, port=args.port, api_key=args.api_key)


# --------------------------------------------------------------------------- models


def cmd_models(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="mlx-dspark models")
    ap.parse_args(argv)
    print("Targets whose drafter auto-resolves — pass any of these as --model and the matched\n"
          "DSpark/DFlash drafter is picked automatically (quantization-agnostic):\n")
    rows = [("target (--model)", "DSpark drafter", "DFlash drafter", "RAM")]
    for e in REGISTRY:
        rows.append((e["target"], e["dspark"], e["dflash"], e["ram"]))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    for i, r in enumerate(rows):
        print("  " + "  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
        if i == 0:
            print("  " + "  ".join("-" * widths[j] for j in range(4)))
    print("\nAny other target works too — pass its HF repo/path as --model plus --drafter <repo>.\n"
          "Use --mode baseline for plain target decoding (no drafter).")


# --------------------------------------------------------------------------- doctor


def cmd_doctor(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="mlx-dspark doctor")
    ap.parse_args(argv)
    ok = True

    def check(label: str, good: bool, detail: str = ""):
        nonlocal ok
        ok = ok and good
        mark = "\033[32m✓\033[0m" if good else "\033[31m✗\033[0m"
        print(f"  {mark} {label}" + (f"  — {detail}" if detail else ""))

    import platform

    print("mlx-dspark doctor\n")
    is_mac = sys.platform == "darwin"
    is_arm = platform.machine() == "arm64"
    check("Apple Silicon (arm64 macOS)", is_mac and is_arm,
          f"{platform.system()} {platform.machine()}")

    # mlx exposes its version on mlx.core, not the top-level package
    for pkg, ver_attr in (("mlx", "mlx.core"), ("mlx_lm", "mlx_lm"), ("mlx_vlm", "mlx_vlm")):
        try:
            __import__(pkg)
            ver = getattr(__import__(ver_attr, fromlist=["__version__"]), "__version__", "?")
            check(f"{pkg} importable", True, ver)
        except Exception as e:  # noqa: BLE001
            check(f"{pkg} importable", False, str(e))

    try:
        import mlx.core as mx  # noqa: F401
        try:
            mx.zeros((2, 2))  # exercise the Metal path
            check("MLX Metal device works", True)
        except Exception as e:  # noqa: BLE001
            check("MLX Metal device works", False, str(e))
    except Exception:
        pass

    total_gb = _total_ram_gb()
    if total_gb:
        check("System RAM", total_gb >= 15,
              f"{total_gb:.0f} GB (gemma4 preset ~15 GB, qwen3 ~8 GB)")

    from . import __version__
    print(f"\nmlx-dspark {__version__} — {'ready' if ok else 'issues above'}.")
    if not ok:
        sys.exit(1)


def _total_ram_gb() -> float | None:
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- dispatch


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    sub = argv[0] if (argv and not argv[0].startswith("-")) else None
    if sub in ("-h", "--help", None) and not argv:
        # bare `mlx-dspark` -> show top-level help
        print(__doc__)
        return
    if sub == "serve":
        return cmd_serve(argv[1:])
    if sub == "models":
        return cmd_models(argv[1:])
    if sub == "doctor":
        return cmd_doctor(argv[1:])
    if sub == "generate":
        return cmd_generate(argv[1:])
    if sub in _SUBCOMMANDS:  # defensive; unreachable
        return
    # No subcommand (flags only) -> legacy flat generate CLI (backward compatible).
    return cmd_generate(argv)


if __name__ == "__main__":
    main()
