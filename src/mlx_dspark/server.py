"""OpenAI-compatible HTTP server for mlx-dspark — serve a DSpark / DFlash / baseline
model over `/v1/chat/completions` so any OpenAI client (LM Studio, the `openai` SDK,
`curl`, LangChain, …) can talk to it locally.

Design choices (deliberate):
  * **Stdlib only.** Built on ``http.server`` (like mlx-lm's own server) so installing
    mlx-dspark stays lean — no FastAPI/uvicorn/pydantic pulled in.
  * **One model, loaded once.** The target + drafter are heavy (~8–15 GB) and load at
    startup; the ``model`` field in a request is echoed back but the loaded pair is always
    used. ``GET /v1/models`` advertises what's loaded.
  * **Serialized generation.** MLX is a single device context and every request builds its
    own KV cache, so generations can't safely interleave — an ``Engine`` lock runs them one
    at a time (correct for a single-user local server; extra requests queue).
  * **Lossless, and it shows.** Whatever the mode, output equals normal decoding of the
    target; the speculative speedup surfaces in a non-standard ``x_mlx_dspark`` block
    (accept length, tok/s) and at ``GET /metrics``.

Endpoints: ``POST /v1/chat/completions`` (stream + non-stream), ``POST /v1/completions``,
``GET /v1/models``, ``GET /health``, ``GET /metrics``.
"""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .generate import (
    GenResult,
    dflash_generate,
    encode_messages,
    greedy_generate,
    speculative_generate,
)
from .load import load_dflash, load_drafter, load_target, resolve
from .prefix_cache import PrefixCache, target_cache_reusable
from .tools import normalize_tool_messages, parse_tool_calls

MODES = ("dspark", "dflash", "baseline")


# --------------------------------------------------------------------------- engine


class Engine:
    """Holds the loaded target/drafter and turns prompt token ids into a GenResult.

    All generation goes through :meth:`generate`, which is guarded by a lock so only one
    request decodes at a time. Cumulative throughput stats are kept for ``/metrics``.
    """

    def __init__(
        self,
        target,
        tokenizer,
        drafter,
        *,
        mode: str,
        model_id: str,
        target_repo: str,
        drafter_repo: str | None,
        max_draft_tokens: int | None,
        confidence_threshold: float = 0.0,
        template_defaults: dict | None = None,
        prefix_cache: bool = True,
        prefix_cache_dir: str | None = None,
        prefix_cache_max_ram_mb: int = 0,
    ):
        self.target = target
        self.tokenizer = tokenizer
        self.drafter = drafter
        self.mode = mode
        self.model_id = model_id
        self.target_repo = target_repo
        self.drafter_repo = drafter_repo
        self.max_draft_tokens = max_draft_tokens
        self.confidence_threshold = confidence_threshold
        # chat-template kwargs applied to every request unless the request overrides them
        # (e.g. {"enable_thinking": False} to silence Qwen3's <think> blocks by default).
        self.template_defaults = dict(template_defaults or {})
        self.prefix = self._build_prefix_cache(
            prefix_cache, prefix_cache_dir, prefix_cache_max_ram_mb)
        # All generation runs on ONE dedicated thread. MLX arrays are thread/stream-affine, so a
        # persistent prefix cache created on one request's thread can't be reused on another's —
        # a single worker keeps every cache create/reuse on the same thread (and serializes work).
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-gen")
        self.created = int(time.time())
        self.stats = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generation_seconds": 0.0,
            "sum_accept_len": 0.0,   # accept-len weighted by tokens, for a token-weighted mean
        }

    def _build_prefix_cache(self, enabled, l2_dir, max_ram_mb):
        """Enable prefix caching only where reuse is exact: dspark/baseline on a dense,
        trimmable (KVCache) target. Disabled for DFlash and for the rotating/sliding-window
        caches Gemma-4 uses — those fall back to a fresh prefill per request."""
        if not enabled or self.mode == "dflash":
            return None
        try:
            if not target_cache_reusable(self.target.make_cache()):
                return None
        except Exception:  # noqa: BLE001
            return None
        make_ctx = self.drafter.make_ctx_cache if self.mode == "dspark" else None
        return PrefixCache(self.target.make_cache, make_ctx,
                           l2_dir=l2_dir, max_ram_bytes=max(0, max_ram_mb) * 1024 * 1024)

    # --- construction ---
    @classmethod
    def load(
        cls,
        *,
        mode: str = "dspark",
        model: str | None = None,
        drafter: str | None = None,
        family: str | None = None,     # deprecated alias for `model`
        target: str | None = None,     # deprecated alias for `model`
        drafter_bits: int = 4,
        max_draft_tokens: int | None = None,
        confidence_threshold: float = 0.0,
        enable_thinking: bool | None = None,
        prefix_cache: bool = True,
        prefix_cache_dir: str | None = None,
        prefix_cache_max_ram_mb: int = 0,
    ) -> "Engine":
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        target_repo, drafter_repo = resolve(model, mode=mode, drafter=drafter,
                                            family=family, target=target)

        tgt, tok = load_target(target_repo)
        draft = None
        if mode == "dspark":
            draft, _ = load_drafter(drafter_repo, quantize=drafter_bits > 0,
                                    bits=max(drafter_bits, 2))
        elif mode == "dflash":
            draft, _ = load_dflash(drafter_repo, quantize=drafter_bits > 0,
                                   bits=max(drafter_bits, 2))
            draft.bind(tgt.model)

        # default cap: dspark's measured optimum is 2; dflash's native point is the full block
        if max_draft_tokens is None and mode == "dspark":
            max_draft_tokens = 2
        model_id = target_repo.rstrip("/").split("/")[-1]
        template_defaults = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        return cls(tgt, tok, draft, mode=mode, model_id=model_id, target_repo=target_repo,
                   drafter_repo=drafter_repo, max_draft_tokens=max_draft_tokens,
                   confidence_threshold=confidence_threshold, template_defaults=template_defaults,
                   prefix_cache=prefix_cache, prefix_cache_dir=prefix_cache_dir,
                   prefix_cache_max_ram_mb=prefix_cache_max_ram_mb)

    # --- generation ---
    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        stop: list[str] | None,
        seed: int | None,
        on_text=None,
    ) -> GenResult:
        # hop onto the single generation thread (keeps all MLX/cache work same-thread)
        return self._executor.submit(
            self._generate_impl, prompt_ids, max_tokens, temperature, top_p, top_k,
            stop, seed, on_text).result()

    def _generate_impl(self, prompt_ids, max_tokens, temperature, top_p, top_k,
                       stop, seed, on_text) -> GenResult:
        # prefix caching: reuse the shared conversation prefix's KV (dspark/baseline on a
        # dense target); `cache is None` means this mode/target doesn't reuse.
        cache = ctx = None
        reuse_len = 0
        if self.prefix is not None:
            cache, ctx, reuse_len = self.prefix.acquire(prompt_ids)
        try:
            if self.mode == "dspark":
                res = speculative_generate(
                    self.target, self.tokenizer, self.drafter, prompt_ids=prompt_ids,
                    cache=cache, ctx_caches=ctx, reuse_len=reuse_len,
                    max_new_tokens=max_tokens, max_draft_tokens=self.max_draft_tokens,
                    confidence_threshold=self.confidence_threshold,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    seed=seed, stop=stop, on_text=on_text,
                )
            elif self.mode == "dflash":
                res = dflash_generate(
                    self.target, self.tokenizer, self.drafter, prompt_ids=prompt_ids,
                    max_new_tokens=max_tokens, max_draft_tokens=self.max_draft_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    seed=seed, stop=stop, on_text=on_text,
                )
            else:
                res = greedy_generate(
                    self.target, self.tokenizer, prompt_ids=prompt_ids,
                    cache=cache, reuse_len=reuse_len,
                    max_new_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    top_k=top_k, seed=seed, stop=stop, on_text=on_text,
                )
        except BaseException:                     # never leave a desynced cache behind
            if self.prefix is not None:
                self.prefix.reset()
            raise
        if self.prefix is not None and cache is not None:
            self.prefix.store(cache, ctx, prompt_ids, res.token_ids)
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += len(prompt_ids)
        self.stats["completion_tokens"] += res.num_tokens
        self.stats["generation_seconds"] += res.seconds
        self.stats["sum_accept_len"] += res.mean_accept_len * res.num_tokens
        return res

    def spec_info(self, res: GenResult) -> dict:
        """The non-standard block we attach so the spec-decode benefit is visible."""
        return {
            "mode": self.mode,
            "accept_len": round(res.mean_accept_len, 3),
            "tokens_per_sec": round(res.tokens_per_sec, 1),
            "target_forwards": res.target_forwards,
        }

    def metrics(self) -> dict:
        s = self.stats
        ct = s["completion_tokens"]
        return {
            "model": self.model_id,
            "mode": self.mode,
            "requests": s["requests"],
            "prompt_tokens": s["prompt_tokens"],
            "completion_tokens": ct,
            "mean_accept_len": round(s["sum_accept_len"] / ct, 3) if ct else 0.0,
            "mean_tokens_per_sec": round(ct / s["generation_seconds"], 1)
            if s["generation_seconds"] else 0.0,
            "prefix_cache": self.prefix.info() if self.prefix is not None else {"enabled": False},
        }


# --------------------------------------------------------------------------- request parsing


def _norm_stop(stop) -> list[str]:
    """OpenAI ``stop`` may be a string, a list, or null -> always a list[str]."""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [str(s) for s in stop]


def _clamp_tokens(v, default: int = 512) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, 8192))


# --------------------------------------------------------------------------- HTTP handler


def make_handler(engine: Engine, api_key: str | None):
    """Build a request-handler class bound to this engine (needed since BaseHTTPRequestHandler
    is instantiated per-connection by the server and can't take extra constructor args)."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "mlx-dspark"

        # -- low-level replies --
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send_json(self, status: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, message: str, etype: str = "invalid_request_error"):
            self._send_json(status, {"error": {"message": message, "type": etype,
                                               "code": status}})

        def _sse_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

        def _sse(self, obj: dict):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
            self.wfile.flush()

        # -- auth --
        def _authed(self) -> bool:
            if not api_key:
                return True
            auth = self.headers.get("Authorization", "")
            return auth == f"Bearer {api_key}"

        def log_message(self, fmt, *args):  # quieter default logging
            return

        # -- routing --
        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                return self._send_json(200, {"status": "ok", "model": engine.model_id,
                                             "mode": engine.mode})
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._send_json(200, self._models_payload())
            if self.path.rstrip("/") == "/metrics":
                return self._send_json(200, engine.metrics())
            return self._send_error(404, f"unknown route {self.path}", "not_found")

        def do_POST(self):
            if not self._authed():
                return self._send_error(401, "invalid api key", "authentication_error")
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                return self._send_error(400, f"invalid JSON body: {e}")

            route = self.path.rstrip("/")
            try:
                if route in ("/v1/chat/completions", "/chat/completions"):
                    return self._chat(req)
                if route in ("/v1/completions", "/completions"):
                    return self._completions(req)
            except (BrokenPipeError, ConnectionResetError):
                return  # client hung up mid-stream; nothing more to do
            except Exception as e:  # keep the server alive on a bad request
                return self._send_error(500, f"generation failed: {e}", "server_error")
            return self._send_error(404, f"unknown route {self.path}", "not_found")

        # -- payloads --
        def _models_payload(self) -> dict:
            return {
                "object": "list",
                "data": [{
                    "id": engine.model_id,
                    "object": "model",
                    "created": engine.created,
                    "owned_by": "mlx-dspark",
                    "x_mlx_dspark": {"mode": engine.mode, "target": engine.target_repo,
                                     "drafter": engine.drafter_repo},
                }],
            }

        def _chat(self, req: dict):
            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                return self._send_error(400, "'messages' must be a non-empty list")
            # chat-template kwargs: server defaults, then per-request overrides. Supports the
            # common `chat_template_kwargs` extension and a top-level `enable_thinking` shortcut.
            tkw = {**engine.template_defaults, **(req.get("chat_template_kwargs") or {})}
            if "enable_thinking" in req:
                tkw["enable_thinking"] = bool(req["enable_thinking"])
            if req.get("tools"):                      # let the template render the tool schemas
                tkw["tools"] = req["tools"]
            try:
                prompt_ids = encode_messages(
                    engine.tokenizer, normalize_tool_messages(messages), **tkw)
            except Exception as e:
                return self._send_error(400, f"could not apply chat template: {e}")
            self._run(req, prompt_ids, chat=True)

        def _completions(self, req: dict):
            prompt = req.get("prompt")
            if isinstance(prompt, list):  # OpenAI allows a batch; we take the first
                prompt = prompt[0] if prompt else ""
            if not isinstance(prompt, str):
                return self._send_error(400, "'prompt' must be a string")
            prompt_ids = list(engine.tokenizer.encode(prompt))
            self._run(req, prompt_ids, chat=False)

        def _run(self, req: dict, prompt_ids: list[int], *, chat: bool):
            params = dict(
                max_tokens=_clamp_tokens(req.get("max_tokens") or req.get("max_completion_tokens")),
                temperature=float(req.get("temperature", 0.0) or 0.0),
                top_p=float(req.get("top_p", 1.0) if req.get("top_p") is not None else 1.0),
                top_k=int(req.get("top_k", 0) or 0),
                stop=_norm_stop(req.get("stop")),
                seed=req.get("seed"),
            )
            model = req.get("model") or engine.model_id
            stream = bool(req.get("stream", False))
            want_tools = bool(chat and req.get("tools"))
            cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
            created = int(time.time())

            if stream:
                return self._run_stream(prompt_ids, params, model, cid, created, chat,
                                        req, want_tools)

            res = engine.generate(prompt_ids, on_text=None, **params)
            usage = {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": res.num_tokens,
                "total_tokens": len(prompt_ids) + res.num_tokens,
            }
            if chat:
                content, finish, tool_calls = res.text, res.finish_reason, None
                if want_tools:
                    parsed, cleaned = parse_tool_calls(res.text)
                    if parsed:
                        tool_calls, content, finish = parsed, (cleaned or None), "tool_calls"
                message = {"role": "assistant", "content": content}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                choice = {"index": 0, "message": message, "finish_reason": finish}
                obj = {"id": cid, "object": "chat.completion", "created": created,
                       "model": model, "choices": [choice], "usage": usage,
                       "x_mlx_dspark": engine.spec_info(res)}
            else:
                choice = {"index": 0, "text": res.text, "finish_reason": res.finish_reason}
                obj = {"id": cid, "object": "text_completion", "created": created,
                       "model": model, "choices": [choice], "usage": usage,
                       "x_mlx_dspark": engine.spec_info(res)}
            self._send_json(200, obj)

        def _run_stream(self, prompt_ids, params, model, cid, created, chat, req, want_tools):
            self._sse_start()
            obj_type = "chat.completion.chunk" if chat else "text_completion"

            def base(delta_or_text, finish):
                if chat:
                    ch = {"index": 0, "delta": delta_or_text, "finish_reason": finish}
                else:
                    ch = {"index": 0, "text": delta_or_text, "finish_reason": finish}
                return {"id": cid, "object": obj_type, "created": created,
                        "model": model, "choices": [ch]}

            # opening chunk announces the assistant role (chat only)
            if chat:
                self._sse(base({"role": "assistant"}, None))

            if want_tools:
                # buffer, then emit tool_calls (or cleaned content) in one delta — incremental
                # tool-call streaming isn't reliable to reconstruct, so we resolve at the end
                res = engine.generate(prompt_ids, on_text=None, **params)
                parsed, cleaned = parse_tool_calls(res.text)
                if parsed:
                    self._sse(base({"tool_calls": [{"index": i, **tc}
                                                   for i, tc in enumerate(parsed)]}, None))
                    finish = "tool_calls"
                else:
                    if cleaned:
                        self._sse(base({"content": cleaned}, None))
                    finish = res.finish_reason
            else:
                def on_text(piece: str):
                    self._sse(base({"content": piece} if chat else piece, None))
                res = engine.generate(prompt_ids, on_text=on_text, **params)
                finish = res.finish_reason

            # final chunk carries finish_reason (+ usage if the client asked for it)
            final = base({} if chat else "", finish)
            opts = req.get("stream_options") or {}
            if opts.get("include_usage"):
                final["usage"] = {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": res.num_tokens,
                    "total_tokens": len(prompt_ids) + res.num_tokens,
                }
            final["x_mlx_dspark"] = engine.spec_info(res)
            self._sse(final)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


# --------------------------------------------------------------------------- entrypoint


def run_server(engine: Engine, *, host: str = "127.0.0.1", port: int = 8080,
               api_key: str | None = None) -> None:
    handler = make_handler(engine, api_key)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    base = f"http://{host}:{port}"
    print("=" * 64)
    print(f"  mlx-dspark server  ·  mode={engine.mode}  ·  model={engine.model_id}")
    print(f"  target : {engine.target_repo}")
    if engine.drafter_repo:
        print(f"  drafter: {engine.drafter_repo}")
    if engine.prefix is not None:
        print(f"  prefix cache: on{'  (+SSD spill)' if engine.prefix.l2_dir else ''}")
    else:
        print("  prefix cache: off (not reusable for this mode/target)")
    print(f"  listening on {base}   (OpenAI base_url: {base}/v1)")
    if api_key:
        print("  auth   : Bearer <api-key> required")
    print("=" * 64)
    print(f"  curl {base}/v1/chat/completions -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"model\":\"{engine.model_id}\",\"messages\":"
          "[{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true}'")
    print("=" * 64, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        httpd.server_close()
