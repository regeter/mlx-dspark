"""Protocol tests for the OpenAI-compatible server.

These use a *mock* engine (no model weights), so they run in CI in milliseconds and
verify the HTTP surface: routing, JSON shapes, SSE framing, stop handling wiring, auth,
and error paths. End-to-end correctness with a real drafter is exercised separately.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from mlx_dspark.generate import GenResult
from mlx_dspark import server as S


class _FakeTok:
    def encode(self, text):
        return [ord(c) for c in text][:64]


class _FakeEngine:
    mode = "dspark"
    model_id = "FakeModel"
    created = 123
    target_repo = "org/Target"
    drafter_repo = "org/Drafter"
    template_defaults = {}

    def __init__(self):
        self.tokenizer = _FakeTok()
        self.calls = []
        self.response_text = "Hello world from mlx dspark"

    def generate(self, prompt_ids, *, max_tokens, temperature, top_p=1.0, top_k=0,
                 stop, seed, on_text=None):
        self.calls.append(dict(prompt_ids=prompt_ids, max_tokens=max_tokens,
                               temperature=temperature, top_p=top_p, top_k=top_k,
                               stop=stop, seed=seed))
        text = self.response_text
        if on_text:
            for w in text.split(" "):
                on_text(w + " ")
        return GenResult(text=text, token_ids=[1, 2, 3, 4, 5], num_tokens=5, num_rounds=2,
                         accept_lengths=[2, 3], target_forwards=2, seconds=0.1,
                         finish_reason="stop")

    def spec_info(self, res):
        return {"mode": self.mode, "accept_len": res.mean_accept_len,
                "tokens_per_sec": res.tokens_per_sec, "target_forwards": res.target_forwards}

    def metrics(self):
        return {"model": self.model_id, "mode": self.mode, "requests": len(self.calls)}


@pytest.fixture
def server():
    eng = _FakeEngine()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, api_key=None))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield eng, f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _get(base, path):
    return json.loads(urllib.request.urlopen(base + path).read())


def _post(base, path, obj, stream=False, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), headers=h, method="POST")
    r = urllib.request.urlopen(req)
    return r.read().decode() if stream else json.loads(r.read())


def test_health(server):
    _, base = server
    h = _get(base, "/health")
    assert h["status"] == "ok" and h["model"] == "FakeModel" and h["mode"] == "dspark"


def test_models(server):
    _, base = server
    m = _get(base, "/v1/models")
    assert m["object"] == "list"
    assert m["data"][0]["id"] == "FakeModel"
    assert m["data"][0]["x_mlx_dspark"]["mode"] == "dspark"


def test_chat_non_stream(server):
    eng, base = server
    c = _post(base, "/v1/chat/completions",
              {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert c["object"] == "chat.completion"
    assert c["choices"][0]["message"]["content"] == "Hello world from mlx dspark"
    assert c["choices"][0]["finish_reason"] == "stop"
    assert c["usage"]["completion_tokens"] == 5 and c["usage"]["prompt_tokens"] > 0
    assert c["usage"]["total_tokens"] == c["usage"]["prompt_tokens"] + 5
    assert "x_mlx_dspark" in c


def test_chat_stream_sse(server):
    _, base = server
    sse = _post(base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}], "stream": True,
                 "stream_options": {"include_usage": True}}, stream=True)
    lines = [l for l in sse.split("\n\n") if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines if l != "data: [DONE]"]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert all(ch["object"] == "chat.completion.chunk" for ch in chunks)
    content = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks)
    assert content == "Hello world from mlx dspark "
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 5


def test_stop_forwarded(server):
    eng, base = server
    _post(base, "/v1/chat/completions",
          {"messages": [{"role": "user", "content": "hi"}], "stop": "END", "temperature": 0.7})
    assert eng.calls[-1]["stop"] == ["END"]
    assert eng.calls[-1]["temperature"] == 0.7


def test_completions_legacy(server):
    _, base = server
    lc = _post(base, "/v1/completions", {"prompt": "once upon"})
    assert lc["object"] == "text_completion"
    assert lc["choices"][0]["text"]
    assert lc["choices"][0]["finish_reason"] == "stop"


_TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def test_tool_calls_non_stream(server):
    eng, base = server
    eng.response_text = 'ok<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'
    c = _post(base, "/v1/chat/completions",
              {"messages": [{"role": "user", "content": "call f"}], "tools": _TOOLS})
    msg = c["choices"][0]["message"]
    assert c["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "f"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


def test_tool_calls_stream(server):
    eng, base = server
    eng.response_text = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    sse = _post(base, "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "call f"}], "tools": _TOOLS,
                 "stream": True}, stream=True)
    chunks = [json.loads(l[6:]) for l in sse.split("\n\n")
              if l.startswith("data: ") and l != "data: [DONE]"]
    tc = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert tc and tc[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_no_tools_means_plain_text(server):
    eng, base = server
    eng.response_text = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    # without `tools` in the request we do NOT parse tool calls — return raw text
    c = _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert c["choices"][0]["message"].get("tool_calls") is None
    assert "<tool_call>" in c["choices"][0]["message"]["content"]


def test_metrics(server):
    _, base = server
    _post(base, "/v1/completions", {"prompt": "x"})
    mt = _get(base, "/metrics")
    assert mt["model"] == "FakeModel" and mt["requests"] >= 1


def test_unknown_route_404(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base, "/nope")
    assert e.value.code == 404


def test_bad_chat_body_400(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/v1/chat/completions", {"messages": []})
    assert e.value.code == 400


def test_auth_required():
    eng = _FakeEngine()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, api_key="secret"))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        assert e.value.code == 401
        # with the right key it works
        c = _post(base, "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "hi"}]},
                  headers={"Authorization": "Bearer secret"})
        assert c["object"] == "chat.completion"
    finally:
        httpd.shutdown()
