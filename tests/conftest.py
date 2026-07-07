"""Shared test setup. On machines without a Metal GPU (e.g. virtualized CI runners),
fall back to the CPU device so the model-free suite still runs — every test here is
numerics-light and device-agnostic."""

import mlx.core as mx

if not mx.metal.is_available():
    mx.set_default_device(mx.cpu)
