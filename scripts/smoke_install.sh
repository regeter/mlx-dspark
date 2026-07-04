#!/usr/bin/env bash
# Clean-install smoke test — run before publishing a release.
#
# WHY THIS EXISTS: our dev .venv is frozen at whatever dependency versions it was
# created with, but a new user's `pip install mlx-dspark` resolves *transitive* deps
# (transformers, tokenizers, ...) to their newest compatible versions — which can be
# newer than .venv and break at import time. That exact gap shipped once: .venv sat on
# transformers 5.12.1 while fresh installs pulled 5.13, which crashed `import mlx_dspark`
# (issue #1). Testing only .venv could never have caught it.
#
# This builds a FRESH environment that resolves deps the way a new user would, then does
# the thing that has broken before — `import mlx_dspark` — and prints the resolved
# versions so drift is visible. It downloads NO model weights. (First run is slow: it
# installs the whole mlx stack; uv caches wheels so re-runs are fast.)
#
# Usage:
#   scripts/smoke_install.sh            # fresh install + import smoke
#   scripts/smoke_install.sh --tests    # also run the model-free test suite in the fresh env
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
VENV="$TMP/smoke-venv"
RUN_TESTS=0
[[ "${1:-}" == "--tests" ]] && RUN_TESTS=1

cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "==> Creating fresh venv: $VENV"
uv venv "$VENV" --python 3.12 >/dev/null

echo "==> Installing mlx-dspark from source (resolving latest deps, like a new user)"
uv pip install --python "$VENV/bin/python" "$ROOT" >/dev/null

echo "==> Resolved dependency versions (watch for drift vs .venv):"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
for pkg in ("mlx", "mlx-lm", "mlx-vlm", "mlx-audio", "transformers", "tokenizers"):
    try:
        print(f"    {pkg:12} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"    {pkg:12} (not installed)")
PY

echo "==> Import smoke (the thing that has broken before):"
"$VENV/bin/python" - <<'PY'
import mlx_dspark
# exercise the full import chain that pulls mlx_lm / mlx_vlm at module scope
from mlx_dspark import load_pair, speculative_generate, Engine  # noqa: F401
print(f"    import mlx_dspark OK — version {mlx_dspark.__version__}")
PY

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "==> Running model-free test suite in the fresh env:"
  uv pip install --python "$VENV/bin/python" pytest >/dev/null
  "$VENV/bin/python" -m pytest "$ROOT/tests" -q
fi

echo "==> Smoke test PASSED"
