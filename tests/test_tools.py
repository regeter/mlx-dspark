"""Unit tests for tool-call parsing (both model formats) and inbound normalization."""

from __future__ import annotations

import json

from mlx_dspark.tools import normalize_tool_messages, parse_tool_calls


def test_hermes_single():
    text = 'Sure.<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}\n</tool_call>'
    tcs, cleaned = parse_tool_calls(text)
    assert len(tcs) == 1
    assert tcs[0]["type"] == "function" and tcs[0]["id"].startswith("call_")
    assert tcs[0]["function"]["name"] == "get_weather"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"city": "Paris", "unit": "c"}
    assert cleaned == "Sure."


def test_hermes_multiple():
    text = ('<tool_call>{"name":"a","arguments":{}}</tool_call>'
            '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>')
    tcs, _ = parse_tool_calls(text)
    assert [t["function"]["name"] for t in tcs] == ["a", "b"]
    assert json.loads(tcs[1]["function"]["arguments"]) == {"x": 1}


def test_hermes_malformed_json_skipped():
    tcs, cleaned = parse_tool_calls('<tool_call>{not json}</tool_call>tail')
    assert tcs == [] and "tail" in cleaned


def test_gemma_scalars_and_strings():
    text = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,unit:<|"|>c<|"|>,days:3,live:true}<tool_call|>'
    tcs, cleaned = parse_tool_calls(text)
    assert len(tcs) == 1 and tcs[0]["function"]["name"] == "get_weather"
    assert json.loads(tcs[0]["function"]["arguments"]) == {
        "city": "Paris", "unit": "c", "days": 3, "live": True}
    assert cleaned == ""


def test_gemma_comma_inside_string():
    text = '<|tool_call>call:say{msg:<|"|>hello, world<|"|>}<tool_call|>'
    tcs, _ = parse_tool_calls(text)
    assert json.loads(tcs[0]["function"]["arguments"]) == {"msg": "hello, world"}


def test_no_tool_call():
    tcs, cleaned = parse_tool_calls("just a normal answer")
    assert tcs == [] and cleaned == "just a normal answer"


def test_normalize_arguments_string_to_dict():
    msgs = [{"role": "assistant", "content": None,
             "tool_calls": [{"type": "function",
                             "function": {"name": "f", "arguments": '{"a": 1}'}}]}]
    n = normalize_tool_messages(msgs)
    assert n[0]["content"] == ""                       # null content coerced to ""
    assert n[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}


def test_normalize_leaves_plain_messages():
    n = normalize_tool_messages([{"role": "user", "content": "hi"}])
    assert n[0] == {"role": "user", "content": "hi"}


def test_normalize_list_content_to_string():
    # OpenAI structured-content parts — what Pi/Continue/Cline & OpenAI SDKs send.
    # A list reaching the chat template errors as "'list object' has no attribute 'startswith'".
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Hello there"}]}]
    n = normalize_tool_messages(msgs)
    assert n[0]["content"] == "Hello there"


def test_normalize_list_content_multipart_drops_nontext():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "line1"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
        {"type": "text", "text": "line2"},
    ]}]
    n = normalize_tool_messages(msgs)
    assert n[0]["content"] == "line1\nline2"


def test_normalize_empty_list_content():
    n = normalize_tool_messages([{"role": "user", "content": []}])
    assert n[0]["content"] == ""
