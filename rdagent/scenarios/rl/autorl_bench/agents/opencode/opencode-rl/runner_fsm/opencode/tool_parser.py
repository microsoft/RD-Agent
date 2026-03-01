from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_-]+)?\n(?P<body>.*?)\n```", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<(?P<tag>bash|read|edit)>(?P<body>.*?)</(?P=tag)>", re.DOTALL | re.IGNORECASE)
_ATTR_TAG_START_RE = re.compile(r"<(?P<tag>bash|read|write|edit)\b", re.IGNORECASE)
_ATTR_NAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_-]*")
_INLINE_TOOL_JSON_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<tag>bash|read|write|edit)\s*(?P<json>\{)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolCall:
    kind: str  # bash | file
    start: int
    payload: dict[str, Any] | str


@dataclass(frozen=True)
class ToolResult:
    kind: str
    ok: bool
    detail: dict[str, Any]


def _xml_unescape(text: str) -> str:
    """Unescape common XML/HTML entities in tool-call payloads."""
    s = str(text or "")
    for _ in range(3):
        s2 = html.unescape(s)
        if s2 == s:
            break
        s = s2
    return s


def _decode_attr_value(raw: str) -> str:
    """Decode minimal escapes inside XML-like attribute values."""
    out: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt in {'"', "'", "\\"}:
                out.append(nxt)
            else:
                out.append("\\")
                out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_attrs(attrs_raw: str) -> dict[str, str]:
    """Parse XML-like key/value attributes with quoted values."""
    attrs: dict[str, str] = {}
    i = 0
    n = len(attrs_raw)
    while i < n:
        while i < n and attrs_raw[i].isspace():
            i += 1
        if i >= n:
            break

        km = _ATTR_NAME_RE.match(attrs_raw, i)
        if not km:
            i += 1
            continue
        key = km.group(0)
        i = km.end()

        while i < n and attrs_raw[i].isspace():
            i += 1
        if i >= n or attrs_raw[i] != "=":
            continue
        i += 1

        while i < n and attrs_raw[i].isspace():
            i += 1
        if i >= n:
            break

        quote = attrs_raw[i]
        if quote not in {'"', "'"}:
            start = i
            while i < n and not attrs_raw[i].isspace():
                i += 1
            attrs[key] = attrs_raw[start:i]
            continue
        i += 1

        buf: list[str] = []
        while i < n:
            ch = attrs_raw[i]
            if ch == "\\" and i + 1 < n:
                buf.append(ch)
                buf.append(attrs_raw[i + 1])
                i += 2
                continue
            if ch == quote:
                i += 1
                break
            buf.append(ch)
            i += 1

        attrs[key] = _decode_attr_value("".join(buf))
    return attrs


def _extract_attr_loose(attrs_raw: str, key: str) -> str | None:
    """Best-effort attribute extractor for malformed/self-closing tags."""
    km = re.search(rf"\b{re.escape(key)}\s*=\s*(['\"])", attrs_raw, flags=re.IGNORECASE)
    if not km:
        return None
    quote = km.group(1)
    start = km.end()
    if start >= len(attrs_raw):
        return None

    if key.lower() == "content":
        end = attrs_raw.rfind(quote)
        if end <= start:
            return None
        return _decode_attr_value(attrs_raw[start:end])

    i = start
    escaped = False
    while i < len(attrs_raw):
        ch = attrs_raw[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == quote:
            return _decode_attr_value(attrs_raw[start:i])
        i += 1
    return None


def _repair_json_escapes(s: str) -> str:
    """Fix invalid JSON escape sequences inside double-quoted strings."""
    valid_next = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    out: list[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(s):
        ch = s[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            if i + 1 >= len(s):
                out.append("\\\\")
                i += 1
                continue
            nxt = s[i + 1]
            if nxt in valid_next:
                out.append("\\")
                escaped = True
                i += 1
                continue
            out.append("\\\\")
            i += 1
            continue
        out.append(ch)
        if ch == '"':
            in_string = False
        i += 1
    return "".join(out)


def _repair_json_keys(s: str) -> str:
    """Fix missing opening quotes on JSON keys."""
    return re.sub(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)"\s*:', r'\1"\2":', s)


def _try_json(text: str) -> dict[str, Any] | None:
    s = text.strip()
    if not s:
        return None
    try:
        data = json.loads(s)
    except Exception:
        for repair_fn in (
            lambda t: _repair_json_escapes(t),
            lambda t: _repair_json_keys(t),
            lambda t: _repair_json_escapes(_repair_json_keys(t)),
        ):
            repaired = repair_fn(s)
            if repaired != s:
                try:
                    data = json.loads(repaired)
                    return data if isinstance(data, dict) else None
                except Exception:
                    continue
        return None
    return data if isinstance(data, dict) else None


def _extract_json_object(text: str, start: int) -> tuple[str, int] | None:
    """Extract a balanced JSON object from `text[start:]` where `text[start] == '{'`."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1], idx + 1
    return None


def _find_tag_gt(text: str, start: int) -> int:
    """Find the end of an XML-like opening tag, respecting quoted attrs."""
    i = start
    quote: str | None = None
    escaped = False
    while i < len(text):
        ch = text[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == ">":
            return i
        i += 1
    return -1


def _iter_attr_tags(text: str) -> Iterable[tuple[str, int, str, str]]:
    """Yield (tag, start, attrs_raw, body) for `<bash|read|write|edit ...>` tags."""
    lower = text.lower()
    pos = 0
    while True:
        m = _ATTR_TAG_START_RE.search(text, pos)
        if not m:
            break
        tag = (m.group("tag") or "").strip().lower()
        start = m.start()
        gt = _find_tag_gt(text, m.end())
        if gt < 0:
            pos = m.end()
            continue

        raw_open = text[m.end() : gt]
        raw_open_rstrip = raw_open.rstrip()
        self_closing = raw_open_rstrip.endswith("/")
        attrs_raw = raw_open_rstrip[:-1].rstrip() if self_closing else raw_open.strip()

        body = ""
        end = gt + 1
        if not self_closing:
            close = f"</{tag}>"
            close_idx = lower.find(close, gt + 1)
            if close_idx < 0:
                pos = gt + 1
                continue
            body = text[gt + 1 : close_idx]
            end = close_idx + len(close)

        yield tag, start, attrs_raw, body
        pos = end


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []

    text = re.sub(r"(?i)(?<!<)\b(bash|read|write|edit)\s*<", r"<\1 ", text)

    for tag, start, attrs_raw, body in _iter_attr_tags(text):
        attrs = _parse_attrs(attrs_raw)
        if "command" not in attrs:
            cmd = _extract_attr_loose(attrs_raw, "command")
            if cmd is not None:
                attrs["command"] = cmd
        if "filePath" not in attrs:
            fp = _extract_attr_loose(attrs_raw, "filePath")
            if fp is not None:
                attrs["filePath"] = fp
        if tag == "write":
            ct = _extract_attr_loose(attrs_raw, "content")
            if ct is not None:
                prev = attrs.get("content")
                if not isinstance(prev, str) or len(ct) > len(prev):
                    attrs["content"] = ct
        if tag == "edit":
            os_ = _extract_attr_loose(attrs_raw, "oldString")
            if os_ is not None:
                prev = attrs.get("oldString")
                if not isinstance(prev, str) or len(os_) > len(prev):
                    attrs["oldString"] = os_
            ns_ = _extract_attr_loose(attrs_raw, "newString")
            if ns_ is not None:
                prev = attrs.get("newString")
                if not isinstance(prev, str) or len(ns_) > len(prev):
                    attrs["newString"] = ns_

        if tag == "bash" and attrs.get("command"):
            calls.append(
                ToolCall(
                    kind="bash", start=start,
                    payload={"command": attrs.get("command", ""), "description": attrs.get("description", "")},
                )
            )
        if tag == "read" and attrs.get("filePath"):
            calls.append(ToolCall(kind="file", start=start, payload={"filePath": attrs["filePath"]}))
        if tag == "write" and attrs.get("filePath"):
            payload2: dict[str, Any] = {"filePath": attrs["filePath"]}
            if body:
                payload2["content"] = body
            elif "content" in attrs:
                payload2["content"] = attrs["content"].replace("\\n", "\n")
            calls.append(ToolCall(kind="file", start=start, payload=payload2))
        if tag == "edit" and attrs.get("filePath"):
            payload2 = {"filePath": attrs["filePath"]}
            if body:
                payload2["content"] = body
            if "oldString" in attrs:
                payload2["oldString"] = attrs["oldString"].replace("\\n", "\n")
            if "newString" in attrs:
                payload2["newString"] = attrs["newString"].replace("\\n", "\n")
            calls.append(ToolCall(kind="file", start=start, payload=payload2))

    for m in _TOOL_CALL_RE.finditer(text):
        inner = m.group("body") or ""
        for tm in _TAG_RE.finditer(inner):
            tag = (tm.group("tag") or "").strip().lower()
            body = (tm.group("body") or "").strip()
            data = _try_json(body)
            if not data:
                continue
            if tag == "bash" and isinstance(data.get("command"), str):
                calls.append(ToolCall(kind="bash", start=m.start(), payload=data))
            if tag in ("read", "edit") and isinstance(data.get("filePath"), str):
                calls.append(ToolCall(kind="file", start=m.start(), payload=data))

        data2 = _try_json(inner)
        if data2:
            raw_name = data2.get("name") or data2.get("tool") or data2.get("toolName")
            raw_args = data2.get("arguments") or data2.get("args")
            if isinstance(raw_name, str) and isinstance(raw_args, dict):
                name = raw_name.strip().lower()
                args = raw_args
                if name == "bash" and isinstance(args.get("command"), str):
                    calls.append(ToolCall(kind="bash", start=m.start(), payload=args))
                if name in ("read", "write", "edit") and isinstance(args.get("filePath"), str):
                    calls.append(ToolCall(kind="file", start=m.start(), payload=args))
            else:
                if isinstance(data2.get("command"), str):
                    calls.append(ToolCall(kind="bash", start=m.start(), payload=data2))
                if isinstance(data2.get("filePath"), str):
                    calls.append(ToolCall(kind="file", start=m.start(), payload=data2))

    for m in _FENCE_RE.finditer(text):
        lang = (m.group("lang") or "").strip().lower()
        body = (m.group("body") or "").strip()

        if lang == "json":
            data = _try_json(body)
            if data and isinstance(data.get("filePath"), str):
                calls.append(ToolCall(kind="file", start=m.start(), payload=data))
            continue

        if lang == "bash":
            lines = body.splitlines()
            if not lines:
                continue
            if lines[0].strip().lower() != "bash":
                continue
            data = _try_json("\n".join(lines[1:]))
            if data and isinstance(data.get("command"), str):
                calls.append(ToolCall(kind="bash", start=m.start(), payload=data))
            continue

    for m in _INLINE_TOOL_JSON_RE.finditer(text):
        tag = (m.group("tag") or "").strip().lower()
        json_start = int(m.start("json"))
        extracted = _extract_json_object(text, json_start)
        if not extracted:
            continue
        raw_json, _ = extracted
        data = _try_json(raw_json)
        if not data:
            continue
        if tag == "bash" and isinstance(data.get("command"), str):
            calls.append(ToolCall(kind="bash", start=m.start(), payload=data))
        if tag in ("read", "write", "edit") and isinstance(data.get("filePath"), str):
            calls.append(ToolCall(kind="file", start=m.start(), payload=data))

    calls.sort(key=lambda c: c.start)
    merged: list[ToolCall] = []
    for c in calls:
        if merged and merged[-1].start == c.start and merged[-1].kind == c.kind:
            prev = merged[-1]
            prev_size = len(json.dumps(prev.payload, sort_keys=True, ensure_ascii=False))
            cur_size = len(json.dumps(c.payload, sort_keys=True, ensure_ascii=False))
            if cur_size >= prev_size:
                merged[-1] = c
            continue
        merged.append(c)
    calls = merged

    uniq: list[ToolCall] = []
    seen: set[str] = set()
    for c in calls:
        key = json.dumps({"kind": c.kind, "payload": c.payload}, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    for c in uniq:
        if not isinstance(c.payload, dict):
            continue
        for k in ("filePath", "content", "oldString", "newString", "command", "description"):
            v = c.payload.get(k)
            if isinstance(v, str) and v:
                c.payload[k] = _xml_unescape(v)
    return uniq


def format_tool_results(results: list[ToolResult]) -> str:
    payload = [r.detail | {"tool": r.kind, "ok": r.ok} for r in results]
    return (
        "Tool results (executed by the runner). Continue by either issuing more tool calls or responding normally.\n\n"
        "```tool_result\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
