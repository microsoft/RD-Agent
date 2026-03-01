"""Transparent LLM API proxy that logs streaming tokens to a file.

Architecture:
    OpenCode Server  ---->  This Proxy  ---->  Real LLM API
                              |
                              v
                         token log file  ---->  _LLMWaitMonitor tails this

Usage:
    python llm_proxy.py --port 8201 --upstream https://real-api/v1 --log /tmp/tokens.log

The proxy:
1. Forwards all requests to the upstream LLM API unchanged
2. For streaming responses (SSE), intercepts chunks and writes decoded tokens to the log
3. For non-streaming responses, forwards as-is
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.request import Request, urlopen
from urllib.error import HTTPError


_upstream: str = ""
_log_path: str = ""
_debug: bool = False
_upstream_timeout: int = 600

# Persistent log file handle + lock (Fix 10: avoid open/close per write)
_log_lock = threading.Lock()
_log_file = None  # type: ignore


def _open_log():
    """Open the persistent log file handle. Called once from main()."""
    global _log_file
    if _log_path:
        _log_file = open(_log_path, "a", encoding="utf-8")


def _close_log():
    """Close the persistent log file handle."""
    global _log_file
    if _log_file is not None:
        try:
            _log_file.close()
        except Exception:
            pass
        _log_file = None


def _write_log(msg: str):
    with _log_lock:
        try:
            if _log_file is not None:
                _log_file.write(msg)
                _log_file.flush()
        except Exception:
            pass


def _extract_token(chunk: dict) -> str:
    """Extract token content from an SSE chunk, trying multiple formats."""
    # Standard OpenAI format: choices[].delta.content
    for c in chunk.get("choices") or []:
        delta = c.get("delta") or {}
        content = delta.get("content")
        if content:
            return content
        # Some models put text in "text" instead of "content"
        text = delta.get("text")
        if text:
            return text
        # Reasoning/thinking content
        reasoning = delta.get("reasoning_content")
        if reasoning:
            return reasoning

    # Non-standard: top-level content
    if chunk.get("content"):
        return chunk["content"]

    # Anthropic-style: content[].text
    for block in chunk.get("content") or []:
        if isinstance(block, dict) and block.get("text"):
            return block["text"]

    return ""


def _log_tool_event(chunk: dict):
    """从 SSE chunk 中提取 server 端工具调用事件，写入 token log。

    支持两种格式：
    1. Responses API:
       - output_item.added: 工具名（立即反馈）
       - output_item.done: 工具名 + 完整参数（详细信息）
    2. Chat Completions API: tool_calls delta
    """
    event_type = chunk.get("type", "")

    # Responses API: added 事件 → 立即显示工具名
    if event_type == "response.output_item.added":
        item = chunk.get("item") or {}
        if item.get("type") == "function_call":
            name = item.get("name", "unknown")
            _write_log(f"\n[TOOL] {name}\n")
            return

    # Responses API: done 事件 → 解析参数，显示详细信息
    if event_type == "response.output_item.done":
        item = chunk.get("item") or {}
        if item.get("type") == "function_call":
            name = item.get("name", "unknown")
            args_str = item.get("arguments", "")
            detail, content_lines = _extract_tool_detail_full(name, args_str)
            if not detail and args_str:
                detail = args_str[:150].replace("\n", " ").strip()
                if len(args_str) > 150:
                    detail += "..."
            if detail:
                _write_log(f"\n[TOOL_DETAIL] {name}: {detail}\n")
            # 写入额外内容行（文件内容预览、patch diff 等）
            for cl in content_lines:
                _write_log(f"[TOOL_CONTENT] {cl}\n")
            return

    # Chat Completions API 格式
    for c in chunk.get("choices") or []:
        delta = c.get("delta") or {}
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function") or {}
            name = func.get("name")
            if name:
                _write_log(f"\n[TOOL] {name}\n")
                return


def _extract_tool_detail_full(name: str, args_str: str) -> tuple[str, list[str]]:
    """从工具调用参数中提取详情和内容预览。

    返回 (summary_line, content_lines)。
    summary_line 是一行摘要，content_lines 是额外的内容预览行（最多 15 行）。
    """
    detail = _extract_tool_detail(name, args_str)
    content_lines: list[str] = []

    if not args_str:
        return detail, content_lines

    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        args = None

    if isinstance(args, dict):
        if name in ("write",):
            # 显示文件内容预览（前 15 行）
            content = args.get("content", "")
            if content:
                lines = content.splitlines()
                for line in lines[:15]:
                    content_lines.append(line[:150])
                if len(lines) > 15:
                    content_lines.append(f"... ({len(lines)} lines total)")

        elif name == "edit":
            # 显示旧代码 → 新代码
            old = args.get("old_string", "") or args.get("oldString", "")
            new = args.get("new_string", "") or args.get("newString", "")
            if old or new:
                for line in old.splitlines()[:5]:
                    content_lines.append(f"- {line[:120]}")
                if len(old.splitlines()) > 5:
                    content_lines.append(f"  ... ({len(old.splitlines())} lines)")
                for line in new.splitlines()[:5]:
                    content_lines.append(f"+ {line[:120]}")
                if len(new.splitlines()) > 5:
                    content_lines.append(f"  ... ({len(new.splitlines())} lines)")

        elif name == "apply_patch":
            # 显示 diff 预览（支持 patchText / patch / diff 多种 key）
            patch = (args.get("patch", "") or args.get("patchText", "")
                     or args.get("diff", "") or "")
            if patch:
                lines = patch.splitlines()
                for line in lines[:15]:
                    content_lines.append(line[:150])
                if len(lines) > 15:
                    content_lines.append(f"... ({len(lines)} lines total)")

    return detail, content_lines


def _extract_tool_detail(name: str, args_str: str) -> str:
    """从工具调用参数中提取关键信息用于展示。"""
    if not args_str:
        return ""
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        # apply_patch 等工具参数可能不是 JSON
        # 尝试提取文件路径
        if "---" in args_str and "+++" in args_str:
            for line in args_str.splitlines():
                if line.startswith("+++ "):
                    path = line[4:].strip()
                    if path.startswith("b/"):
                        path = path[2:]
                    return path[:120]
        return args_str[:80].replace("\n", " ").strip()

    if isinstance(args, dict):
        if name == "bash":
            cmd = args.get("command", "")
            if not cmd:
                return ""
            # 对短命令（<= 200 字符）完整显示（将换行替换为 ; ）
            if len(cmd) <= 200:
                return cmd.replace("\n", " ; ").strip()[:200]
            # 较长命令：显示前 3 行，用 ; 连接
            lines = [l.strip() for l in cmd.split("\n") if l.strip()]
            preview = " ; ".join(lines[:3])
            if len(lines) > 3:
                preview += " ..."
            return preview[:200]
        elif name == "read":
            return args.get("filePath", "") or args.get("file_path", "")
        elif name in ("write", "edit"):
            return args.get("filePath", "") or args.get("file_path", "")
        elif name in ("glob", "Glob"):
            return args.get("pattern", "") or str(args)[:80]
        elif name in ("grep", "Grep"):
            pat = args.get("pattern", "")
            path = args.get("path", "")
            if pat and path:
                return f"{pat} in {path}"
            return pat or str(args)[:80]
        elif name == "apply_patch":
            patch = (args.get("patch", "") or args.get("patchText", "")
                     or args.get("diff", "") or str(args))
            # 提取所有被修改的文件路径
            files = []
            for pline in patch.splitlines():
                if pline.startswith("+++ "):
                    path = pline[4:].strip()
                    if path.startswith("b/"):
                        path = path[2:]
                    files.append(path)
                # OpenCode *** Add File / *** Update File 格式
                elif pline.startswith("*** Add File: "):
                    files.append(pline[14:].strip())
                elif pline.startswith("*** Update File: "):
                    files.append(pline[17:].strip())
            if files:
                return ", ".join(f[:80] for f in files[:3])
            # Fallback: 显示 patch 内容摘要
            return patch[:100].replace("\n", " ").strip()
        elif name == "todowrite":
            todos = args.get("todos", [])
            if isinstance(todos, list) and todos:
                first = todos[0]
                if isinstance(first, dict):
                    return first.get("content", "")[:100]
            return ""
        else:
            # 通用 fallback: 返回第一个有意义的字符串值
            for v in args.values():
                if isinstance(v, str) and v.strip() and len(v) < 200:
                    return v.strip()[:100]
            return ""
    return ""


def _parse_sse_line(text: str) -> str | None:
    """Parse an SSE data line, return the payload or None if not a data line."""
    # Standard: "data: {...}" or "data: [DONE]"
    if text.startswith("data: "):
        return text[6:]
    # No-space variant: "data:{...}"
    if text.startswith("data:"):
        return text[5:]
    return None


def _strip_encrypted(obj: dict | list, _depth: int = 0) -> bool:
    """Recursively strip ``encrypted_content`` fields from a request payload.

    OpenAI Codex models return encrypted reasoning tokens tied to a specific
    organization.  When the conversation history is sent back through a
    different org / Azure deployment, the server rejects them with
    ``invalid_encrypted_content``.  Removing these fields allows multi-turn
    conversations to work across different endpoints.

    Returns True if any field was removed.
    """
    if _depth > 20:
        return False
    changed = False
    if isinstance(obj, dict):
        if "encrypted_content" in obj:
            del obj["encrypted_content"]
            changed = True
        # Also strip reasoning items that only carry encrypted content
        # from Responses API "input" arrays.
        if obj.get("type") == "reasoning" and "encrypted_content" not in obj:
            # Already stripped above; mark for removal from parent list
            pass
        for v in obj.values():
            if isinstance(v, (dict, list)):
                changed |= _strip_encrypted(v, _depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                changed |= _strip_encrypted(item, _depth + 1)
        # Remove reasoning items that are now empty after stripping
        before = len(obj)
        obj[:] = [
            item for item in obj
            if not (isinstance(item, dict) and item.get("type") == "reasoning"
                    and "encrypted_content" not in item
                    and not item.get("summary"))
        ]
        if len(obj) != before:
            changed = True
    return changed


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward requests to upstream, intercept streaming responses."""

    def log_message(self, format, *args):
        # Suppress default access log to stderr
        pass

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len else b""

        # Build upstream request
        url = f"{_upstream}{self.path}"
        headers = {}
        for key in ("Content-Type", "Authorization", "Accept"):
            val = self.headers.get(key)
            if val:
                headers[key] = val
        headers["Content-Type"] = self.headers.get("Content-Type", "application/json")

        # Check if request asks for streaming
        is_stream = False
        try:
            req_json = json.loads(body)
            is_stream = req_json.get("stream", False)
            model = req_json.get("model", "?")
            # Strip encrypted_content from requests to avoid cross-org
            # decryption failures in multi-turn conversations.
            if _strip_encrypted(req_json):
                body = json.dumps(req_json).encode("utf-8")
                headers["Content-Type"] = "application/json"
        except Exception:
            model = "?"

        req = Request(url, data=body, headers=headers, method="POST")

        try:
            resp = urlopen(req, timeout=_upstream_timeout)
        except HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())
            return
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy error: {e}".encode())
            return

        # Forward response headers
        self.send_response(resp.status)
        for k, v in resp.headers.items():
            if k.lower() not in ("transfer-encoding", "connection"):
                self.send_header(k, v)
        self.end_headers()

        if not is_stream:
            data = resp.read()
            self.wfile.write(data)
            return

        # Streaming: read SSE chunks, log tokens, forward to client.
        # Use a reader thread + queue so heartbeats can be emitted even when
        # the upstream LLM is silent (e.g. during extended reasoning/thinking).
        _write_log(f"\n[{time.strftime('%H:%M:%S')}] >>> stream start model={model}\n")
        token_count = 0
        raw_lines = 0
        t0 = time.time()
        _last_heartbeat = t0

        _SENTINEL = None  # signals reader thread is done

        line_q: queue.Queue = queue.Queue(maxsize=256)

        def _reader():
            """Background thread: blocking readline() → queue."""
            try:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    line_q.put(line)
            except Exception as exc:
                line_q.put(exc)
            finally:
                line_q.put(_SENTINEL)

        reader_t = threading.Thread(target=_reader, daemon=True)
        reader_t.start()

        try:
            while True:
                try:
                    item = line_q.get(timeout=10.0)
                except queue.Empty:
                    # No data from upstream in 10s — emit heartbeat
                    _now = time.time()
                    _write_log(f"\n[HEARTBEAT] {raw_lines} chunks {_now - t0:.0f}s\n")
                    _last_heartbeat = _now
                    continue

                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item

                line = item
                # Forward raw bytes to OpenCode server
                self.wfile.write(line)
                self.wfile.flush()

                # Heartbeat on data arrival too (if interval elapsed)
                _now = time.time()
                if _now - _last_heartbeat >= 10.0:
                    _write_log(f"\n[HEARTBEAT] {raw_lines} chunks {_now - t0:.0f}s\n")
                    _last_heartbeat = _now

                # Parse SSE data lines for token content
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                payload = _parse_sse_line(text)
                if payload is None:
                    # Not an SSE data line — log for debugging on first few
                    if _debug and raw_lines < 3:
                        _write_log(f"[DBG non-data] {text[:200]}\n")
                    continue

                raw_lines += 1
                if payload.strip() == "[DONE]":
                    continue

                try:
                    chunk = json.loads(payload)
                    content = _extract_token(chunk)
                    if content:
                        token_count += 1
                        _write_log(content)
                    else:
                        # 提取 server 内部工具调用事件，写入 token log 供 monitor 展示进度
                        _log_tool_event(chunk)
                        if _debug and raw_lines <= 3:
                            _write_log(f"[DBG no-content] {payload[:300]}\n")
                except json.JSONDecodeError:
                    if _debug and raw_lines <= 3:
                        _write_log(f"[DBG bad-json] {payload[:300]}\n")
        except Exception as e:
            _write_log(f"\n[ERR] {e}\n")

        elapsed = time.time() - t0
        _write_log(f"\n[{time.strftime('%H:%M:%S')}] <<< stream end {token_count} tokens {elapsed:.1f}s (data_lines={raw_lines})\n")

    def do_GET(self):
        # Forward GET requests (e.g. /v1/models)
        url = f"{_upstream}{self.path}"
        headers = {}
        for key in ("Authorization", "Accept"):
            val = self.headers.get(key)
            if val:
                headers[key] = val
        req = Request(url, headers=headers, method="GET")
        try:
            resp = urlopen(req, timeout=30)
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.read())
        except HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"proxy error: {e}".encode())


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", type=str, required=True)
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--debug", action="store_true", help="Log raw SSE data for debugging")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Upstream request timeout in seconds (default: 600)")
    args = parser.parse_args()

    global _upstream, _log_path, _debug, _upstream_timeout
    _upstream = args.upstream.rstrip("/")
    _log_path = args.log
    _debug = args.debug
    _upstream_timeout = max(30, args.timeout)

    _open_log()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler)
        print(f"LLM proxy listening on 127.0.0.1:{args.port} -> {_upstream} (timeout={_upstream_timeout}s)", flush=True)
        server.serve_forever()
    finally:
        _close_log()


if __name__ == "__main__":
    main()
