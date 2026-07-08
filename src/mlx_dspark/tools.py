"""Tool-calling glue: translate between OpenAI ``tool_calls`` and each model's native syntax.

Two output formats are parsed (detected by markers, so it doesn't matter which model is
loaded — it also covers any model that borrows one of these conventions):

  * **Hermes / JSON** (Qwen3, DeepSpec drafters' targets, many others)::

        <tool_call>{"name": "f", "arguments": {...}}</tool_call>

  * **Gemma-4** (bespoke)::

        <|tool_call>call:f{key:<|"|>str val<|"|>,n:3,flag:true}<tool_call|>

    where string values are wrapped in Gemma's ``<|"|>`` quote markers and other scalars are
    bare. Flat arguments are parsed fully; deeply nested structures fall back to string values
    (rare for tool calls, and documented).

:func:`parse_tool_calls` returns OpenAI ``tool_calls`` (``function.arguments`` serialized to a
JSON string, per the OpenAI schema) plus the assistant text with the call blocks removed.
:func:`normalize_tool_messages` goes the other way for inbound history: it turns an OpenAI
assistant message's ``function.arguments`` JSON *string* back into a dict so the model's chat
template (which iterates a mapping) renders prior tool calls correctly.
"""

from __future__ import annotations

import json
import re
import uuid

_HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_GEMMA = re.compile(r"<\|tool_call>\s*call:\s*([^\{]+?)\s*\{(.*?)\}\s*<tool_call\|>", re.DOTALL)
_GEMMA_STR = '<|"|>'
_GEMMA_FIELD = re.compile(r'([A-Za-z_]\w*)\s*:\s*(<\|"\|>.*?<\|"\|>|[^,]*)', re.DOTALL)


def _coerce(raw: str):
    raw = raw.strip()
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_gemma_args(body: str) -> dict:
    args: dict = {}
    n = len(_GEMMA_STR)
    for m in _GEMMA_FIELD.finditer(body):
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith(_GEMMA_STR) and raw.endswith(_GEMMA_STR) and len(raw) >= 2 * n:
            args[key] = raw[n:-n]
        else:
            args[key] = _coerce(raw)
    return args


def _as_openai(name: str, args) -> dict:
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": args}}


def parse_tool_calls(text: str) -> tuple[list[dict], str]:
    """(tool_calls, cleaned_text). ``tool_calls`` is OpenAI-shaped; empty if none found."""
    calls: list[tuple[str, object]] = []
    cleaned = text
    if "<tool_call>" in text:
        for m in _HERMES.finditer(text):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("name"):
                calls.append((obj["name"], obj.get("arguments", {})))
        cleaned = _HERMES.sub("", cleaned)
    if "<|tool_call>" in text:
        for m in _GEMMA.finditer(text):
            calls.append((m.group(1).strip(), _parse_gemma_args(m.group(2))))
        cleaned = _GEMMA.sub("", cleaned)
    return [_as_openai(name, args) for name, args in calls], cleaned.strip()


def _flatten_content(content):
    """OpenAI allows ``content`` to be a list of typed parts rather than a plain string
    (``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``) — this is what many
    coding agents / OpenAI SDKs send. The text targets we serve (and their chat templates)
    expect a string; a list reaches the template unchanged and blows up inside it as
    ``'list object' has no attribute 'startswith'``. So join the text parts (dropping
    non-text parts — images/audio have no place in the text path) and hand the template a
    string. A plain string / ``None`` is returned unchanged.
    """
    if not isinstance(content, list):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            parts.append(p["text"])
        elif isinstance(p, str):
            parts.append(p)
    return "\n".join(parts)


def normalize_tool_messages(messages: list[dict]) -> list[dict]:
    """Make an OpenAI message history renderable by the model chat templates:

      * list-valued ``content`` (OpenAI structured content parts) -> concatenated text
        (see :func:`_flatten_content`);
      * assistant ``function.arguments`` JSON strings -> dicts (templates iterate a mapping);
      * ``content: null`` -> ``""`` (OpenAI allows null content on tool-call messages, but the
        Qwen3 / Gemma-4 templates assume a string and error on ``None``).
    """
    out = []
    for m in messages:
        m = dict(m)
        if "content" in m:
            m["content"] = _flatten_content(m["content"])
        if m.get("content", "") is None:
            m["content"] = ""
        tcs = m.get("tool_calls")
        if tcs:
            new = []
            for tc in tcs:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a)
                    except (json.JSONDecodeError, TypeError):
                        pass
                tc["function"] = fn
                new.append(tc)
            m["tool_calls"] = new
        out.append(m)
    return out
