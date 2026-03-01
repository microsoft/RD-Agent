from __future__ import annotations

import base64
import json
import os
import signal
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..dtypes import AgentClient, AgentResult, TurnEvent
from .tool_parser import parse_tool_calls, format_tool_results
from .tool_executor import ToolPolicy, execute_tool_calls
from ..utils.subprocess import tail

# Module-level tracking of all active OpenCodeClient instances for cleanup on signal
_active_clients: set = set()


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Find a free port with SO_REUSEADDR to reduce TOCTOU race window."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        _, port = s.getsockname()
    return port

def select_bash_mode(*, purpose: str, default_bash_mode: str, scaffold_bash_mode: str) -> str:
    p = str(purpose or "").strip().lower()
    default = str(default_bash_mode or "restricted").strip().lower() or "restricted"
    scaffold = str(scaffold_bash_mode or default).strip().lower() or default
    if p in ("scaffold_contract", "repair_contract"):
        return scaffold
    return default

@dataclass(frozen=True)
class OpenCodeServerConfig:

    base_url: str
    username: str
    password: str

class OpenCodeRequestError(RuntimeError):

    def __init__(self, *, method: str, url: str, status: int | None, detail: str):
        super().__init__(f"OpenCode request failed: {method} {url} ({status}) {detail}")
        self.method = method
        self.url = url
        self.status = status
        self.detail = detail

class StaleThinkingTimeout(RuntimeError):
    """Raised when the LLM has been thinking without progress for too long."""
    pass

class OpenCodeClient(AgentClient):

    def __init__(
        self,
        *,
        repo: Path,
        plan_rel: str,
        pipeline_rel: str | None,
        model: str,
        base_url: str | None,
        timeout_seconds: int,
        request_retry_attempts: int = 2,
        request_retry_backoff_seconds: float = 2.0,
        session_recover_attempts: int | None = None,
        session_recover_backoff_seconds: float | None = None,
        context_length: int | None = None,
        max_prompt_chars: int | None = None,
        bash_mode: str,
        scaffold_bash_mode: str = "full",
        unattended: str,
        max_turns: int = 20,
        server_log_path: Path | None = None,
        username: str | None = None,
        password: str | None = None,
        session_title: str | None = None,
        stale_timeout: float = 180.0,
        auto_compact: bool | None = None,
        permission_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._repo = repo
        self._plan_rel = str(plan_rel or "PLAN.md").strip() or "PLAN.md"
        self._pipeline_rel = str(pipeline_rel).strip() if pipeline_rel else None
        self._timeout_seconds = int(timeout_seconds) if timeout_seconds else 300
        self._request_retry_attempts = max(0, int(request_retry_attempts or 0))
        try:
            _backoff = float(request_retry_backoff_seconds)
        except Exception:
            _backoff = 2.0
        self._request_retry_backoff_seconds = max(0.0, _backoff)
        _recover_attempts_raw = (
            session_recover_attempts
            if session_recover_attempts is not None
            else os.environ.get("OPENCODE_SESSION_RECOVER_ATTEMPTS", "2")
        )
        try:
            self._session_recover_attempts = max(0, int(_recover_attempts_raw or 0))
        except Exception:
            self._session_recover_attempts = 2
        _recover_backoff_raw = (
            session_recover_backoff_seconds
            if session_recover_backoff_seconds is not None
            else os.environ.get("OPENCODE_SESSION_RECOVER_BACKOFF_SECONDS", "2.0")
        )
        try:
            self._session_recover_backoff_seconds = max(0.0, float(_recover_backoff_raw or 0.0))
        except Exception:
            self._session_recover_backoff_seconds = 2.0
        try:
            _context_length = int(context_length or 0)
        except Exception:
            _context_length = 0
        self._context_length: int | None = _context_length if _context_length > 0 else None
        try:
            _max_prompt_chars = int(max_prompt_chars or 0)
        except Exception:
            _max_prompt_chars = 0
        self._max_prompt_chars: int | None = _max_prompt_chars if _max_prompt_chars > 0 else None
        self._bash_mode = (bash_mode or "restricted").strip().lower()
        if self._bash_mode not in ("restricted", "full"):
            raise ValueError("invalid_bash_mode")
        self._scaffold_bash_mode = (scaffold_bash_mode or "full").strip().lower()
        if self._scaffold_bash_mode not in ("restricted", "full"):
            raise ValueError("invalid_scaffold_bash_mode")
        self._max_turns = max(1, int(max_turns or 20))
        self._unattended = str(unattended or "strict").strip().lower() or "strict"
        self._session_title = str(session_title or f"runner:{repo.name}")
        self._server_log_path = server_log_path.resolve() if server_log_path is not None else None
        self._permission_overrides = permission_overrides or {}

        model_str = str(model or "").strip()
        if not model_str:
            provider_id, model_id = "openai", "gpt-4o-mini"
        elif "/" in model_str:
            provider_id, model_id = model_str.split("/", 1)
            provider_id = provider_id.strip() or "openai"
            model_id = model_id.strip() or "gpt-4o-mini"
        else:
            provider_id, model_id = "openai", model_str
        self._model_obj: dict[str, str] = {"providerID": provider_id, "modelID": model_id}
        self._model_str: str = f"{provider_id}/{model_id}"

        self._stale_timeout = max(0.0, float(stale_timeout or 180.0))
        self._auto_compact = auto_compact

        self._proc: subprocess.Popen[str] | None = None
        self._proxy_proc: subprocess.Popen[str] | None = None
        self._proxy_log_file = None
        self._token_log_path: Path | None = None
        self._temp_config_home: Path | None = None
        self._server_log_file = None
        self._owns_local_server = not bool(base_url)
        _active_clients.add(self)

        if base_url:
            base_url_s = str(base_url).strip()
            if not base_url_s:
                raise ValueError("empty_url")
            self._server = OpenCodeServerConfig(
                base_url=base_url_s.rstrip("/"),
                username=(username or "opencode").strip() or "opencode",
                password=(password or "").strip(),
            )
        else:
            self._server = self._start_local_server(
                repo=repo, server_log_path=self._server_log_path, username=username
            )

        try:
            deadline = time.time() + 60
            last_err = ""
            _health_start = time.time()
            _health_dots = 0
            while time.time() < deadline:
                try:
                    self._request_json("GET", "/global/health", body=None, require_auth=bool(self._server.password))
                    elapsed_h = time.time() - _health_start
                    print(f"    server ready ({elapsed_h:.1f}s)", flush=True)
                    break
                except Exception as e:
                    last_err = str(e)
                    _health_dots += 1
                    if _health_dots % 25 == 0:  # 每 5 秒打印一次（0.2s * 25）
                        elapsed_h = time.time() - _health_start
                        print(f"    waiting for server... ({elapsed_h:.0f}s)", flush=True)
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"OpenCode server failed health check: {tail(last_err, 2000)}")

            if self._auto_compact is not None:
                try:
                    self._request_json(
                        "PATCH", "/global/config",
                        body={"compaction": {"auto": bool(self._auto_compact)}},
                        require_auth=bool(self._server.password),
                    )
                except Exception:
                    pass  # old server versions may not support this endpoint

            data = self._request_json(
                "POST",
                "/session",
                body={"title": self._session_title},
                require_auth=bool(self._server.password),
            )
            if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].strip():
                self._session_id = data["id"]
            elif isinstance(data, dict) and isinstance(data.get("sessionID"), str) and data["sessionID"].strip():
                self._session_id = data["sessionID"]
            else:
                raise RuntimeError(
                    f"unexpected /session response: {tail(json.dumps(data, ensure_ascii=False), 2000)}"
                )
        except Exception:
            # If init fails after starting a local server, ensure we don't leak the process.
            try:
                self.close()
            except Exception:
                pass
            raise

    def close(self) -> None:
        _active_clients.discard(self)
        self._stop_local_server()
        if self._server_log_file is not None:
            try:
                self._server_log_file.close()
            except Exception:
                pass
            self._server_log_file = None

    @staticmethod
    def _extract_context_info(msg: Any) -> dict[str, Any] | None:
        """Extract token usage and compaction events from a message response.

        Returns a dict with keys: input, output, reasoning, cache_read, cache_write,
        total, cost, compaction (bool), summary (bool).
        Returns None if the response doesn't contain token info (graceful degradation).
        """
        if not isinstance(msg, dict):
            return None
        info = msg.get("info")
        if not isinstance(info, dict):
            return None
        tokens = info.get("tokens")
        if not isinstance(tokens, dict):
            return None

        try:
            # Support nested cache format: {cache: {read: N, write: N}} or flat {cacheRead: N}
            cache_obj = tokens.get("cache")
            if isinstance(cache_obj, dict):
                cache_r = int(cache_obj.get("read") or 0)
                cache_w = int(cache_obj.get("write") or 0)
            else:
                cache_r = int(tokens.get("cacheRead") or tokens.get("cache_read") or 0)
                cache_w = int(tokens.get("cacheWrite") or tokens.get("cache_write") or 0)
            result: dict[str, Any] = {
                "input": int(tokens.get("input") or 0),
                "output": int(tokens.get("output") or 0),
                "reasoning": int(tokens.get("reasoning") or 0),
                "cache_read": cache_r,
                "cache_write": cache_w,
                "total": int(tokens.get("total") or 0),
                "cost": float(tokens.get("cost") or 0.0),
            }
        except (TypeError, ValueError):
            return None

        # If total wasn't provided, compute it
        if result["total"] <= 0:
            result["total"] = result["input"] + result["output"] + result["reasoning"]

        # Check for compaction events in parts
        compaction = False
        summary = False
        parts = msg.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    if part.get("type") == "compaction":
                        compaction = True
                    if part.get("type") == "summary":
                        compaction = True
                        summary = True
        # Also check info.summary flag
        if info.get("summary"):
            summary = True

        result["compaction"] = compaction
        result["summary"] = summary
        return result

    def _print_context_status(self, turn: int, ctx_info: dict[str, Any], session_total: int) -> None:
        """Print [context] and [compact] status lines after each turn."""
        rc = self._LLMWaitMonitor._try_rich()
        inp = ctx_info.get("input", 0)
        out = ctx_info.get("output", 0)

        # Format numbers with commas for readability
        def _fmt(n: int) -> str:
            return f"{n:,}"

        # Build context line
        ctx_len = self._context_length
        if ctx_len and ctx_len > 0:
            pct = (inp / ctx_len * 100) if ctx_len > 0 else 0
            ctx_line = (
                f"[context] turn {turn}: {_fmt(inp)}/{_fmt(ctx_len)} ({pct:.0f}%)"
                f" | out={_fmt(out)} | session={_fmt(session_total)}"
            )
            style = "bold yellow" if pct > 70 else "dim"
        else:
            ctx_line = (
                f"[context] turn {turn}: in={_fmt(inp)} out={_fmt(out)}"
                f" | session total={_fmt(session_total)}"
            )
            style = "dim"

        if rc:
            # Escape [ ] so Rich doesn't eat them as markup tags
            safe = ctx_line.replace("[", "\\[")
            rc.print(f"    [{style}]{safe}[/]")
        else:
            print(f"    {ctx_line}", flush=True)

        # Compaction events
        if ctx_info.get("compaction"):
            if rc:
                rc.print(f"    [bold magenta]\\[compact] auto-compaction triggered[/]")
            else:
                print(f"    [compact] auto-compaction triggered", flush=True)
        if ctx_info.get("summary"):
            if rc:
                rc.print(f"    [bold magenta]\\[compact] summary message (post-compaction)[/]")
            else:
                print(f"    [compact] summary message (post-compaction)", flush=True)

    def _stop_local_server(self) -> None:
        if self._proc is not None:
            already_dead = self._proc.poll() is not None
            if not already_dead:
                try:
                    # Do not block shutdown on a potentially wedged server.
                    self._request_json("POST", "/instance/dispose", body=None, require_auth=True, timeout_seconds=5)
                except Exception:
                    pass
                try:
                    if os.name == "posix":
                        try:
                            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                        except Exception:
                            self._proc.terminate()
                    else:  # pragma: no cover
                        self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5)
                    except Exception:
                        if os.name == "posix":
                            try:
                                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                            except Exception:
                                self._proc.kill()
                        else:  # pragma: no cover
                            self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._server_log_file is not None:
            try:
                self._server_log_file.close()
            except Exception:
                pass
            self._server_log_file = None
        # Stop LLM proxy
        if self._proxy_proc is not None:
            try:
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(self._proxy_proc.pid), signal.SIGTERM)
                    except Exception:
                        self._proxy_proc.terminate()
                else:
                    self._proxy_proc.terminate()
                self._proxy_proc.wait(timeout=3)
            except Exception:
                try:
                    self._proxy_proc.kill()
                except Exception:
                    pass
            finally:
                self._proxy_proc = None
        if self._proxy_log_file is not None:
            try:
                self._proxy_log_file.close()
            except Exception:
                pass
            self._proxy_log_file = None
        # Clean up temp config dir
        if self._temp_config_home is not None:
            try:
                shutil.rmtree(self._temp_config_home, ignore_errors=True)
            except Exception:
                pass
            self._temp_config_home = None

    class _LLMWaitMonitor:
        """Tail token log (from LLM proxy) + heartbeat while waiting for LLM response.

        Token log format (written by llm_proxy.py):
            [HH:MM:SS] >>> stream start model=glm-4.7
            <raw token content, no newlines between tokens>
            [HH:MM:SS] <<< stream end 523 tokens 4.2s

        Output format:
            | >>> stream start model=glm-4.7
            | 让我先看看数据格式，用 terminal 执行以下命令：...
            | <<< stream end 523 tokens 4.2s
        """

        _DEFAULT_STALE_TIMEOUT = 180.0

        def __init__(self, token_log: Path | None, turn: int,
                     heartbeat_interval: float = 5.0,
                     stale_timeout: float = 180.0,
                     server_proc=None):
            self._token_log = token_log
            self._turn = turn
            self._interval = heartbeat_interval
            self._stale_timeout = stale_timeout
            self._server_proc = server_proc  # 超时时杀进程触发重试
            self._client_ref = None  # set externally to flag stale timeout on client
            self._stop = threading.Event()
            self._timed_out = threading.Event()
            self._thread: threading.Thread | None = None
            self._start = 0.0
            self._log_offset = 0
            self._line_buf = ""  # accumulate tokens into lines
            # Running token count from proxy stream-end lines
            self._proxy_output_tokens = 0
            self._proxy_rounds = 0
            # Streaming character count for real-time estimation (~4 chars/token)
            self._proxy_stream_chars = 0

        def start(self):
            self._start = time.time()
            if self._token_log:
                try:
                    self._log_offset = self._token_log.stat().st_size
                except Exception:
                    self._log_offset = 0
                try:
                    from pipeline.ui import console as _rc
                    _rc.print(f"    [dim italic]... waiting for LLM response (turn {self._turn})[/]")
                except Exception:
                    print(f"    ... waiting for LLM response (turn {self._turn})", flush=True)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def stop(self):
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=2)
                if self._thread.is_alive():
                    print("    [diag] WARNING: LLM monitor thread did not exit within 2s", flush=True)
            # Flush any remaining buffer
            if self._line_buf.strip():
                text = self._line_buf.strip()
                if len(text) > 200:
                    text = text[:197] + "..."
                self._print_token_line(text)
                self._line_buf = ""

        @staticmethod
        def _try_rich():
            """Get rich console if available."""
            try:
                from pipeline.ui import console as _rc
                return _rc
            except Exception:
                return None

        def _estimate_tokens(self) -> int:
            """Best available token estimate: exact (from stream end) or char-based."""
            exact = self._proxy_output_tokens
            estimated = self._proxy_stream_chars // 4  # ~4 chars per token
            return max(exact, estimated)

        def _token_suffix(self) -> str:
            """Build a token/round suffix string for display.

            Three states:
            - Tokens available: "| ~N tokens / R rounds"
            - Stream in progress but no visible tokens (reasoning model): "| round R streaming"
            - No stream activity: ""
            """
            est = self._estimate_tokens()
            if est > 0:
                suffix = f" | ~{est:,} tokens"
                if self._proxy_rounds > 1:
                    suffix += f" / {self._proxy_rounds} rounds"
                return suffix
            # Stream started but no content yet (reasoning model thinking)
            if self._proxy_rounds > 0:
                if self._proxy_rounds > 1:
                    return f" | round {self._proxy_rounds} streaming"
                return " | streaming"
            return ""

        def _print_thinking(self, elapsed: float):
            rc = self._try_rich()
            suffix = self._token_suffix()
            if rc:
                rc.print(f"    [dim italic]... LLM thinking ({elapsed:.0f}s){suffix}[/]")
            else:
                print(f"    ... LLM thinking ({elapsed:.0f}s){suffix}", flush=True)

        def _print_tool_call(self, tool_name: str, elapsed: float):
            rc = self._try_rich()
            suffix = self._token_suffix()
            if rc:
                rc.print(f"    [bold cyan]... agent calling: {tool_name}[/] [dim]({elapsed:.0f}s){suffix}[/]")
            else:
                print(f"    ... agent calling: {tool_name} ({elapsed:.0f}s){suffix}", flush=True)

        def _print_tool_detail(self, detail: str):
            if len(detail) > 200:
                detail = detail[:197] + "..."
            rc = self._try_rich()
            if rc:
                rc.print(f"    [cyan]    {detail}[/]")
            else:
                print(f"        {detail}", flush=True)

        def _print_tool_content(self, text: str):
            """Print tool content preview line (file content, diff, etc.)."""
            if len(text) > 150:
                text = text[:147] + "..."
            rc = self._try_rich()
            if rc:
                # diff 着色
                if text.startswith("+") and not text.startswith("+++"):
                    rc.print(f"    [green]      {text}[/]")
                elif text.startswith("-") and not text.startswith("---"):
                    rc.print(f"    [red]      {text}[/]")
                elif text.startswith("@@") or text.startswith("---") or text.startswith("+++"):
                    rc.print(f"    [cyan]      {text}[/]")
                elif text.startswith("*** "):
                    rc.print(f"    [bold cyan]      {text}[/]")
                else:
                    rc.print(f"    [dim]      {text}[/]")
            else:
                print(f"          {text}", flush=True)

        def _print_token_line(self, text: str):
            if len(text) > 200:
                text = text[:197] + "..."
            rc = self._try_rich()
            if rc:
                rc.print(f"    [dim]| {text}[/]")
            else:
                print(f"    | {text}", flush=True)

        def is_timed_out(self) -> bool:
            """Whether the monitor triggered a stale-thinking timeout."""
            return self._timed_out.is_set()

        def _run(self):
            last_activity = time.time()
            last_real_activity = time.time()  # 只在有真实内容时更新
            while not self._stop.wait(0.3):
                printed = self._tail_token_log()
                now = time.time()
                if printed:
                    last_activity = now
                    last_real_activity = now
                elif (now - last_activity) >= self._interval:
                    elapsed = now - self._start
                    stale = now - last_real_activity
                    # 检测 LLM 长时间无进展
                    if self._stale_timeout and stale > self._stale_timeout:
                        rc = self._try_rich()
                        msg = (f"    LLM stale for {stale:.0f}s "
                               f"(>{self._stale_timeout:.0f}s), aborting to retry...")
                        if rc:
                            rc.print(f"    [bold yellow]{msg}[/]")
                        else:
                            print(f"    {msg}", flush=True)
                        self._timed_out.set()
                        # Signal client to NOT recover — propagate the error
                        if self._client_ref is not None:
                            self._client_ref._stale_timeout_event.set()
                        # 杀掉 server 进程以中断阻塞的 HTTP 请求
                        if self._server_proc:
                            try:
                                self._server_proc.kill()
                            except Exception:
                                pass
                        break
                    self._print_thinking(elapsed)
                    last_activity = now

        def _tail_token_log(self) -> bool:
            if not self._token_log:
                return False
            try:
                size = self._token_log.stat().st_size
                if size <= self._log_offset:
                    return False
                with open(self._token_log, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(self._log_offset)
                    new = f.read(size - self._log_offset)
                self._log_offset = size
                if not new:
                    return False
            except Exception:
                return False

            printed = False
            # Count raw streaming chars for real-time token estimation.
            # Exclude control lines ([TOOL], [DBG], stream markers) — only
            # actual LLM output text contributes to the estimate.
            for ch in new:
                if ch == "\n":
                    line = self._line_buf.strip()
                    self._line_buf = ""
                    if not line:
                        continue
                    # Parse proxy stream markers for running token count
                    if "<<< stream" in line:
                        # Format: [HH:MM:SS] <<< stream end 523 tokens 4.2s
                        try:
                            parts_s = line.split()
                            for i, w in enumerate(parts_s):
                                if w == "tokens" and i > 0:
                                    self._proxy_output_tokens += int(parts_s[i - 1])
                                    break
                        except (ValueError, IndexError):
                            pass
                        printed = True  # count as activity
                        continue
                    if ">>> stream" in line:
                        self._proxy_rounds += 1
                        printed = True  # count as activity to delay stale timeout
                        continue
                    # Heartbeat from proxy (reasoning models produce no visible tokens)
                    if line.startswith("[HEARTBEAT]"):
                        printed = True  # keep stale timeout at bay
                        continue
                    # 过滤噪音
                    if ("[DBG" in line
                            or '"encrypted_content"' in line
                            or ('"type":"response.' in line and len(line) > 150)):
                        continue
                    # [TOOL_CONTENT] → 工具内容预览行（缩进显示）
                    if line.startswith("[TOOL_CONTENT] "):
                        content = line[15:]
                        self._print_tool_content(content)
                        printed = True
                        continue
                    # [TOOL_DETAIL] → 详细信息（缩进显示在 [TOOL] 下方）
                    if line.startswith("[TOOL_DETAIL] "):
                        detail = line[14:].strip()
                        self._print_tool_detail(detail)
                        printed = True
                        continue
                    # [TOOL] → agent 正在调用工具
                    if line.startswith("[TOOL] "):
                        tool_name = line[7:].strip()
                        elapsed = time.time() - self._start
                        self._print_tool_call(tool_name, elapsed)
                        printed = True
                        continue
                    self._proxy_stream_chars += len(line)
                    self._print_token_line(line)
                    printed = True
                else:
                    self._line_buf += ch
                    # Flush partial tokens every 80 chars (show streaming progress)
                    # BUT: [TOOL lines must be processed as complete lines — don't split them
                    buf_len = len(self._line_buf)
                    if buf_len >= 80 and "\n" not in self._line_buf:
                        text = self._line_buf.strip()
                        # Keep accumulating [TOOL / stream marker lines until newline
                        if (text.startswith("[TOOL") or ">>> stream" in text or "<<< stream" in text) and buf_len < 500:
                            pass  # keep accumulating until newline
                        else:
                            if text and "[DBG" not in text:
                                self._proxy_stream_chars += len(text)
                                self._print_token_line(text)
                                printed = True
                            self._line_buf = ""
            return printed

    def run(self, text: str, *, fsm_state: str, iter_idx: int, purpose: str, on_turn=None) -> AgentResult:
        policy = ToolPolicy(
            repo=self._repo.resolve(),
            plan_path=(self._repo / self._plan_rel).resolve(),
            pipeline_path=((self._repo / self._pipeline_rel).resolve() if self._pipeline_rel else None),
            purpose=purpose,
            bash_mode=select_bash_mode(
                purpose=purpose,
                default_bash_mode=self._bash_mode,
                scaffold_bash_mode=self._scaffold_bash_mode,
            ),
            unattended=self._unattended,
        )

        prompt = text
        trace: list[dict[str, Any]] = []
        self._stale_timeout_event = threading.Event()  # reset per run
        cumulative_tokens: dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}
        for turn_idx in range(self._max_turns):
            monitor = None
            if on_turn:
                monitor = self._LLMWaitMonitor(
                    self._token_log_path, turn_idx + 1,
                    stale_timeout=self._stale_timeout,
                    server_proc=self._proc if hasattr(self, '_proc') else None,
                )
                monitor._client_ref = self  # allow monitor to signal stale timeout
            try:
                if monitor:
                    monitor.start()
                try:
                    msg = self._post_message_with_retry(model=self._model_obj, text=prompt)
                except OpenCodeRequestError as e:
                    # Compatibility fallback: some builds may accept model as a string.
                    if e.status in (400, 422):
                        msg = self._post_message_with_retry(model=self._model_str, text=prompt)
                    else:
                        raise
            except StaleThinkingTimeout:
                raise  # propagate stale timeout without recovery
            finally:
                if monitor:
                    monitor.stop()

            # Extract and display context/token info
            ctx_info = self._extract_context_info(msg)
            if ctx_info is not None:
                cumulative_tokens["input"] += ctx_info["input"]
                cumulative_tokens["output"] += ctx_info["output"]
                cumulative_tokens["reasoning"] += ctx_info["reasoning"]
                session_total = cumulative_tokens["input"] + cumulative_tokens["output"] + cumulative_tokens["reasoning"]
                self._print_context_status(turn_idx + 1, ctx_info, session_total)

            opencode_err = None
            if isinstance(msg, dict):
                info = msg.get("info")
                if isinstance(info, dict):
                    err_obj = info.get("error")
                    if isinstance(err_obj, dict):
                        name = str(err_obj.get("name") or "").strip() or "Error"
                        data = err_obj.get("data")
                        detail = ""
                        if isinstance(data, dict):
                            detail = str(data.get("message") or "").strip()
                        if not detail:
                            detail = str(data).strip() if data is not None else ""
                        opencode_err = f"{name}: {detail}" if detail else name
            if opencode_err:
                raise RuntimeError(f"OpenCode agent error: {opencode_err}")

            if not isinstance(msg, dict):
                assistant_text = str(msg)
            else:
                parts = msg.get("parts")
                if not isinstance(parts, list):
                    assistant_text = str(msg)
                else:
                    texts: list[str] = []
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text" and isinstance(part.get("text"), str):
                            t = part["text"]
                            if t.strip():
                                texts.append(t)
                    assistant_text = "\n".join(texts) or str(msg)
            calls = parse_tool_calls(assistant_text)
            if not calls:
                _trace_entry: dict[str, Any] = {
                    "turn": int(turn_idx + 1),
                    "assistant_text_tail": tail(assistant_text or "", 4000),
                    "calls": [],
                    "results": [],
                }
                if ctx_info is not None:
                    _trace_entry["tokens"] = ctx_info
                trace.append(_trace_entry)
                if on_turn:
                    on_turn(TurnEvent(turn=turn_idx + 1, assistant_text=assistant_text, finished=True))
                return AgentResult(assistant_text=assistant_text, raw=msg, tool_trace=trace)

            # 逐个执行工具调用，每完成一个就回调 on_turn，实时显示进展
            results = execute_tool_calls(calls, repo=self._repo, policy=policy)
            compact_results: list[dict[str, Any]] = []
            calls_list = list(calls)
            results_list = list(results)
            for call_idx, r in enumerate(results_list):
                detail = dict(r.detail or {})
                if isinstance(detail.get("content"), str):
                    detail["content"] = tail(detail["content"], 4000)
                if isinstance(detail.get("stdout"), str):
                    detail["stdout"] = tail(detail["stdout"], 4000)
                if isinstance(detail.get("stderr"), str):
                    detail["stderr"] = tail(detail["stderr"], 4000)
                compact_results.append(detail | {"tool": r.kind, "ok": bool(r.ok)})

                # 实时回调：每完成一个工具调用就通知，让 stream printer 显示进展
                if on_turn and call_idx < len(calls_list):
                    on_turn(TurnEvent(
                        turn=turn_idx + 1,
                        assistant_text=assistant_text if call_idx == 0 else "",
                        calls=[calls_list[call_idx]],
                        results=[results_list[call_idx]],
                    ))

            _trace_entry2: dict[str, Any] = {
                "turn": int(turn_idx + 1),
                "assistant_text_tail": tail(assistant_text or "", 4000),
                "calls": [
                    {
                        "kind": str(c.kind),
                        "payload": c.payload if isinstance(c.payload, (dict, list, str, int, float, bool)) else str(c.payload),
                    }
                    for c in calls_list
                ],
                "results": compact_results,
            }
            if ctx_info is not None:
                _trace_entry2["tokens"] = ctx_info
            trace.append(_trace_entry2)

            # For scaffold runs, we don't need the agent to "finish talking" if the contract
            # is already valid. Some models keep emitting extra tool calls indefinitely.
            if str(purpose or "").strip().lower() == "scaffold_contract" and self._pipeline_rel:
                try:
                    from ..core.pipeline_spec import load_pipeline_spec
                    from ..contract.validation import validate_scaffold_contract

                    pipeline_path = (self._repo / self._pipeline_rel).resolve()
                    if pipeline_path.exists():
                        parsed = load_pipeline_spec(pipeline_path)
                        report = validate_scaffold_contract(self._repo, pipeline=parsed, require_metrics=True)
                        if not report.errors:
                            return AgentResult(assistant_text=assistant_text, raw=msg, tool_trace=trace)
                except Exception:
                    pass

            prompt = format_tool_results(results)

        raise RuntimeError(f"OpenCode tool loop exceeded {self._max_turns} turns without a final response.")

    def _start_local_server(
        self,
        *,
        repo: Path,
        server_log_path: Path | None,
        username: str | None,
        append_log: bool = False,
    ) -> OpenCodeServerConfig:
        if not shutil.which("opencode"):
            raise RuntimeError("`opencode` not found in PATH. Install it from https://opencode.ai/")

        host = "127.0.0.1"
        port = _find_free_port(host)

        user = (username or "opencode").strip() or "opencode"
        pwd = secrets.token_urlsafe(24)
        env = dict(os.environ)
        # OpenCode's OpenAI-compatible provider reads `OPENAI_BASE_URL`.
        # Keep compatibility with `.env` files using `OPENAI_API_BASE`.
        if not str(env.get("OPENAI_BASE_URL") or "").strip():
            api_base = str(env.get("OPENAI_API_BASE") or "").strip().rstrip("/")
            if api_base:
                env["OPENAI_BASE_URL"] = api_base if api_base.endswith("/v1") else (api_base + "/v1")
        env["OPENCODE_SERVER_USERNAME"] = user
        env["OPENCODE_SERVER_PASSWORD"] = pwd
        if self._auto_compact is False:
            env["OPENCODE_DISABLE_AUTOCOMPACT"] = "true"

        # --- Start LLM streaming proxy ---
        if server_log_path is not None:
            self._start_llm_proxy(env, server_log_path.parent)

        # If proxy didn't create a temp config but we have permission overrides,
        # create a temp config now to inject them.
        if self._permission_overrides and not hasattr(self, "_temp_config_home"):
            xdg = str(env.get("XDG_CONFIG_HOME") or "").strip()
            cfg_home = Path(xdg) if xdg else Path.home() / ".config"
            cfg_path = cfg_home / "opencode" / "opencode.json"
            if cfg_path.exists():
                import copy
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                perms = cfg.setdefault("permission", {})
                for pk, pv in self._permission_overrides.items():
                    perms[pk] = pv
                self._temp_config_home = Path(tempfile.mkdtemp(prefix="opencode_perm_"))
                tmp_dir = self._temp_config_home / "opencode"
                tmp_dir.mkdir(parents=True)
                (tmp_dir / "opencode.json").write_text(
                    json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8",
                )
                env["XDG_CONFIG_HOME"] = str(self._temp_config_home)

        cmd = ["opencode", "serve", "--hostname", host, "--port", str(port)]

        stdout = subprocess.DEVNULL
        if server_log_path is not None:
            server_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass
            self._server_log_file = server_log_path.open(
                "a" if append_log else "w",
                encoding="utf-8",
            )
            stdout = self._server_log_file

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stdout,
            start_new_session=True,
        )
        return OpenCodeServerConfig(base_url=f"http://{host}:{port}", username=user, password=pwd)

    def _start_llm_proxy(self, env: dict, log_dir: Path) -> None:
        """Start LLM streaming proxy and redirect OpenCode to use it.

        OpenCode reads its LLM API URL from ``~/.config/opencode/opencode.json``
        (via the provider's ``options.baseURL``), **not** from the
        ``OPENAI_BASE_URL`` environment variable.  So we must:

        1. Read the real ``baseURL`` from the OpenCode config.
        2. Start a local proxy pointing at that upstream.
        3. Create a temporary copy of the config with ``baseURL`` pointing
           at the proxy.
        4. Set ``XDG_CONFIG_HOME`` so the OpenCode server process picks up
           the temporary config.
        """
        proxy_script = Path(__file__).with_name("llm_proxy.py")
        if not proxy_script.exists():
            print("    [diag] WARNING: llm_proxy.py not found, token logging and stale timeout visualization disabled", flush=True)
            return

        # ---- 1. Find the real upstream URL ----
        upstream_url = ""
        config_data = None

        # Try OpenCode config (respects XDG_CONFIG_HOME)
        xdg = str(env.get("XDG_CONFIG_HOME") or "").strip()
        config_home = Path(xdg) if xdg else Path.home() / ".config"
        config_path = config_home / "opencode" / "opencode.json"
        provider_id = self._model_obj.get("providerID", "openai")

        if config_path.exists():
            try:
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                providers = config_data.get("provider", {})
                if provider_id in providers:
                    opts = providers[provider_id].get("options", {})
                    upstream_url = str(opts.get("baseURL", "")).strip().rstrip("/")
            except Exception:
                pass

        # Fallback: use OPENAI_BASE_URL env var
        if not upstream_url:
            upstream_url = str(env.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
            # Strip /v1 — the proxy forwards the full path including /v1
            if upstream_url.endswith("/v1"):
                upstream_url = upstream_url[:-3]

        if not upstream_url:
            print("    [diag] WARNING: no upstream URL found for LLM proxy, token logging disabled", flush=True)
            return

        # ---- 2. Start proxy process ----
        proxy_port = _find_free_port("127.0.0.1")

        log_dir.mkdir(parents=True, exist_ok=True)
        self._token_log_path = log_dir / "llm_tokens.log"
        self._token_log_path.write_text("", encoding="utf-8")

        proxy_log = log_dir / "llm_proxy.log"
        self._proxy_log_file = proxy_log.open("w", encoding="utf-8")

        proxy_timeout = int(self._stale_timeout + 300)
        self._proxy_proc = subprocess.Popen(
            [
                sys.executable, str(proxy_script),
                "--port", str(proxy_port),
                "--upstream", upstream_url,
                "--log", str(self._token_log_path),
                "--timeout", str(proxy_timeout),
            ],
            stdin=subprocess.DEVNULL,
            stdout=self._proxy_log_file,
            stderr=self._proxy_log_file,
            text=True,
            start_new_session=True,
        )

        proxy_base = f"http://127.0.0.1:{proxy_port}"

        # ---- 3. Redirect OpenCode to the proxy ----
        # a) If we found an OpenCode config with the provider, create a temp
        #    copy with baseURL pointing at the proxy.
        if config_data and provider_id in config_data.get("provider", {}):
            import copy
            patched = copy.deepcopy(config_data)
            patched["provider"][provider_id]["options"]["baseURL"] = proxy_base
            # Inject permission overrides (key is "permission" singular in OpenCode)
            if self._permission_overrides:
                perms = patched.setdefault("permission", {})
                for perm_key, perm_val in self._permission_overrides.items():
                    perms[perm_key] = perm_val
            self._temp_config_home = Path(tempfile.mkdtemp(prefix="opencode_proxy_"))
            temp_cfg_dir = self._temp_config_home / "opencode"
            temp_cfg_dir.mkdir(parents=True)
            (temp_cfg_dir / "opencode.json").write_text(
                json.dumps(patched, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            env["XDG_CONFIG_HOME"] = str(self._temp_config_home)

        # b) Also set OPENAI_BASE_URL for providers that read it from env.
        env["OPENAI_BASE_URL"] = f"{proxy_base}/v1"

        # Wait for proxy to be ready (poll port instead of blind sleep)
        _proxy_ready = False
        for _pw in range(50):  # up to 10s (50 * 0.2s)
            try:
                with socket.create_connection(("127.0.0.1", proxy_port), timeout=0.5):
                    _proxy_ready = True
                    break
            except OSError:
                time.sleep(0.2)
        if not _proxy_ready:
            print(f"    [diag] WARNING: LLM proxy on port {proxy_port} not ready after 10s, continuing without proxy", flush=True)
            # Kill the proxy process since it's not working
            try:
                self._proxy_proc.kill()
            except Exception:
                pass
            self._proxy_proc = None
            self._token_log_path = None
            return

    def _post_message_with_retry(self, *, model: Any, text: str) -> Any:
        attempts = 1 + int(self._request_retry_attempts or 0)
        include_context = True
        last_err: OpenCodeRequestError | None = None
        recover_budget = int(self._session_recover_attempts or 0) if self._owns_local_server else 0
        recover_tries = 0

        for attempt in range(1, attempts + 1):
            try:
                return self._post_message(model=model, text=text, include_context=include_context)
            except OpenCodeRequestError as e:
                last_err = e
                # Some OpenCode builds may reject unknown fields; degrade gracefully.
                if include_context and self._context_length is not None and e.status in (400, 422):
                    include_context = False
                    try:
                        return self._post_message(model=model, text=text, include_context=False)
                    except OpenCodeRequestError as e2:
                        last_err = e2
                        e = e2

                transport_unavailable = False
                if e.status is None:
                    d = str(e.detail or "").strip().lower()
                    if d:
                        needles = (
                            "connection refused",
                            "failed to establish a new connection",
                            "connection reset",
                            "connection aborted",
                            "connection closed",
                            "remote end closed",
                            "network is unreachable",
                            "name or service not known",
                            "temporary failure in name resolution",
                            "timed out",
                            "timeout",
                        )
                        transport_unavailable = any(n in d for n in needles)

                # If stale timeout killed the server, don't recover — let it propagate
                if getattr(self, '_stale_timeout_event', None) and self._stale_timeout_event.is_set():
                    raise StaleThinkingTimeout(
                        f"LLM stale thinking timeout — server killed by monitor"
                    ) from e

                if transport_unavailable and recover_tries < recover_budget:
                    recover_tries += 1
                    # 诊断：打印触发恢复的原因和 server 进程状态
                    _proc = getattr(self, '_proc', None)
                    _exit_code = _proc.poll() if _proc else "no_proc"
                    print(f"    [diag] transport error: {e.detail}", flush=True)
                    print(f"    [diag] server process exit_code={_exit_code}, recover_try={recover_tries}/{recover_budget}", flush=True)
                    try:
                        recover_fn = getattr(self, "_recover_local_server_session", None)
                        if callable(recover_fn):
                            recover_fn(reason=e.detail)
                        else:
                            if not self._owns_local_server:
                                raise RuntimeError("session_recover_not_local_server")
                            username = (
                                str(getattr(self, "_server", None).username).strip()
                                if getattr(self, "_server", None) is not None
                                else "opencode"
                            ) or "opencode"
                            self._stop_local_server()
                            self._server = self._start_local_server(
                                repo=self._repo,
                                server_log_path=self._server_log_path,
                                username=username,
                                append_log=True,
                            )

                            deadline = time.time() + 60
                            last_health_err = ""
                            _rh_start = time.time()
                            _rh_dots = 0
                            print(f"    recovering server...", flush=True)
                            while time.time() < deadline:
                                try:
                                    self._request_json(
                                        "GET",
                                        "/global/health",
                                        body=None,
                                        require_auth=bool(self._server.password),
                                    )
                                    print(f"    server recovered ({time.time() - _rh_start:.1f}s)", flush=True)
                                    break
                                except Exception as health_exc:
                                    last_health_err = str(health_exc)
                                    _rh_dots += 1
                                    if _rh_dots % 25 == 0:
                                        print(f"    waiting for server... ({time.time() - _rh_start:.0f}s)", flush=True)
                                    time.sleep(0.2)
                            else:
                                raise RuntimeError(
                                    f"OpenCode server failed health check: {tail(last_health_err, 2000)}"
                                )

                            data = self._request_json(
                                "POST",
                                "/session",
                                body={"title": self._session_title},
                                require_auth=bool(self._server.password),
                            )
                            if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].strip():
                                self._session_id = data["id"]
                            elif (
                                isinstance(data, dict)
                                and isinstance(data.get("sessionID"), str)
                                and data["sessionID"].strip()
                            ):
                                self._session_id = data["sessionID"]
                            else:
                                raise RuntimeError(
                                    f"unexpected /session response: {tail(json.dumps(data, ensure_ascii=False), 2000)}"
                                )
                    except Exception as recover_exc:
                        last_err = OpenCodeRequestError(
                            method=e.method,
                            url=e.url,
                            status=e.status,
                            detail=f"{e.detail}; recover_failed: {tail(str(recover_exc), 1200)}",
                        )
                    else:
                        sleep_fn = getattr(self, "_sleep_session_recover_backoff", None)
                        if callable(sleep_fn):
                            sleep_fn(recover_idx=recover_tries)
                        else:
                            base = float(self._session_recover_backoff_seconds or 0.0)
                            if base > 0:
                                delay = min(30.0, base * (2 ** max(0, int(recover_tries) - 1)))
                                if delay > 0:
                                    time.sleep(delay)
                        continue

                should_retry_fn = getattr(self, "_should_retry_request_error", None)
                if callable(should_retry_fn):
                    should_retry = bool(should_retry_fn(e))
                elif e.status is None:
                    should_retry = True
                else:
                    try:
                        code = int(e.status)
                    except Exception:
                        should_retry = True
                    else:
                        should_retry = code in (408, 409, 425, 429) or code >= 500

                if attempt >= attempts or not should_retry:
                    raise

                sleep_fn = getattr(self, "_sleep_retry_backoff", None)
                if callable(sleep_fn):
                    sleep_fn(attempt_idx=attempt)
                else:
                    base = float(self._request_retry_backoff_seconds or 0.0)
                    if base > 0:
                        delay = min(30.0, base * (2 ** max(0, int(attempt) - 1)))
                        if delay > 0:
                            time.sleep(delay)

        if last_err is not None:
            raise last_err
        raise RuntimeError("opencode_retry_failed_without_error")

    def _post_message(self, *, model: Any, text: str, include_context: bool = True) -> Any:
        s = str(text or "")
        cap = self._max_prompt_chars
        if cap is None or cap <= 0 or len(s) <= cap:
            clipped_text = s
        elif cap < 128:
            clipped_text = s[-cap:]
        else:
            marker = "\n...[TRUNCATED_FOR_OPENCODE_CONTEXT]...\n"
            head = max(32, cap // 2)
            tail_keep = max(32, cap - head - len(marker))
            clipped_text = s[:head] + marker + s[-tail_keep:]
        body = {
            "agent": "build",
            "model": model,
            "parts": [{"type": "text", "text": clipped_text}],
        }
        if include_context and self._context_length is not None:
            # Best-effort: different OpenCode versions may ignore this field.
            body["contextLength"] = int(self._context_length)
        # HTTP 超时自动跟随 stale_timeout：让 stale monitor 做真正的超时控制，
        # HTTP 层只是兜底安全网，永远不应先于 stale monitor 触发。
        msg_timeout = max(self._timeout_seconds, self._stale_timeout + 120)
        if not getattr(self, '_diag_timeout_printed', False):
            print(f"    [diag] timeout config: http={self._timeout_seconds}s, stale={self._stale_timeout}s, msg_deadline={msg_timeout}s", flush=True)
            self._diag_timeout_printed = True
        data = self._request_json(
            "POST",
            f"/session/{self._session_id}/message",
            body=body,
            require_auth=bool(self._server.password),
            timeout_seconds=msg_timeout,
        )
        # Some OpenCode builds/transports may respond with 200 + empty body; treat it as a transient transport failure
        # so the caller can retry or recover the local session instead of silently returning "None".
        if data is None:
            url = f"{self._server.base_url}/session/{self._session_id}/message"
            raise OpenCodeRequestError(method="POST", url=url, status=None, detail="connection closed: empty_response_body")
        return data

    def _request_json(self, method: str, path: str, *, body: Any, require_auth: bool, timeout_seconds: float | None = None) -> Any:
        url = f"{self._server.base_url}{path}"
        headers = {"Accept": "application/json"}
        if require_auth and self._server.password:
            token = base64.b64encode(
                f"{self._server.username}:{self._server.password}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"

        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = Request(url, method=method, data=data, headers=headers)
        timeout = self._timeout_seconds if timeout_seconds is None else float(timeout_seconds)

        # Use a background thread for the HTTP request so we can check
        # the stale timeout flag and abort early instead of blocking for
        # the full HTTP timeout (300s).
        result_container: list = []  # [value] or []
        error_container: list = []   # [exception] or []
        resp_ref: list = []          # [response] — 用于外部强制关闭

        def _do_request():
            try:
                resp = urlopen(req, timeout=timeout)
                resp_ref.append(resp)
                try:
                    raw = resp.read()
                    if not raw:
                        result_container.append(None)
                    else:
                        result_container.append(json.loads(raw.decode("utf-8", errors="replace")))
                finally:
                    try:
                        resp.close()
                    except Exception:
                        pass
            except Exception as exc:
                error_container.append(exc)

        req_thread = threading.Thread(target=_do_request, daemon=True)
        req_thread.start()

        # Poll every 2s, checking for stale timeout flag
        deadline = time.time() + timeout
        while time.time() < deadline:
            req_thread.join(timeout=2.0)
            if not req_thread.is_alive():
                break
            # Check if stale timeout was triggered — abort immediately
            if getattr(self, '_stale_timeout_event', None) and self._stale_timeout_event.is_set():
                # 强制关闭底层连接，让后台线程的 read() 立即失败退出
                for r in resp_ref:
                    try:
                        r.close()
                    except Exception:
                        pass
                # 等线程退出（server 已被 kill，socket 应该很快 RST）
                req_thread.join(timeout=5.0)
                raise StaleThinkingTimeout(
                    "LLM stale thinking timeout — aborting HTTP request"
                )

        if req_thread.is_alive():
            # Thread still running after deadline — force close and treat as timeout
            for r in resp_ref:
                try:
                    r.close()
                except Exception:
                    pass
            if "/message" in path:
                print(f"    [diag] HTTP polling deadline reached: timeout={timeout}s, url={path}", flush=True)
            raise OpenCodeRequestError(
                method=method, url=url, status=None,
                detail=f"timeout: request exceeded {timeout}s",
            )

        if error_container:
            e = error_container[0]
            if "/message" in path:
                print(f"    [diag] HTTP thread error on message POST: {type(e).__name__}: {e}", flush=True)
            if isinstance(e, HTTPError):
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = str(e)
                raise OpenCodeRequestError(method=method, url=url, status=int(getattr(e, "code", 0) or 0), detail=tail(detail, 2000))
            elif isinstance(e, URLError):
                raise OpenCodeRequestError(method=method, url=url, status=None, detail=str(e))
            elif isinstance(e, (TimeoutError, socket.timeout)):
                raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"timeout: {e}")
            elif isinstance(e, OSError):
                raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"os_error: {e}")
            else:
                raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"error: {e}")

        return result_container[0] if result_container else None
