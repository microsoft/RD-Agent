from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..utils.security import cmd_allowed, looks_interactive, safe_env
from ..utils.subprocess import STDIO_TAIL_CHARS, run_cmd_capture, tail
from .tool_parser import ToolCall, ToolResult, _xml_unescape


def _is_env_like(path: Path) -> bool:
    name = path.name.lower()
    if name == ".env":
        return True
    if name.startswith(".env."):
        return True
    if name.endswith(".env"):
        return True
    if ".env." in name:
        return True
    return False


def _within_root(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def _sanitized_env(*, unattended: str) -> dict[str, str]:
    base: dict[str, str] = {}
    for k in ("PATH", "HOME", "LANG", "LC_ALL", "TERM"):
        v = os.environ.get(k)
        if v:
            base[k] = v
    for k, v in os.environ.items():
        ku = k.upper()
        if ku.endswith("_KEY") or ku.endswith("_TOKEN") or ku.endswith("_PASSWORD") or ku.endswith("_SECRET"):
            continue
        if ku in ("OPENAI_API_KEY", "OPENCODE_SERVER_PASSWORD", "OPENCODE_SERVER_USERNAME"):
            continue
        base[k] = v
    return safe_env(base, {}, unattended=unattended)


def _restricted_bash_allowed(cmd: str, *, repo: Path) -> tuple[bool, str | None]:
    s = cmd.strip()
    if not s:
        return False, "empty_command"

    if any(ch in s for ch in (";", "|", "&", ">", "<")):
        return False, "blocked_shell_metacharacters"

    try:
        argv = shlex.split(s)
    except ValueError:
        return False, "blocked_unparseable_command"
    if not argv:
        return False, "empty_command"

    prog = argv[0]
    args = argv[1:]

    if prog == "ls":
        return True, None

    if prog in ("rg", "grep"):
        return True, None

    if prog == "git":
        if not args:
            return False, "blocked_git_without_subcommand"
        if args[0] not in ("status", "diff", "log", "show"):
            return False, "blocked_git_subcommand"
        return True, None

    if prog == "cat":
        if not args:
            return False, "blocked_cat_without_path"
        for raw in args:
            p = Path(raw)
            if p.is_absolute() or ".." in p.parts:
                return False, "blocked_cat_non_repo_path"
            abs_p = (repo / p).resolve()
            if not _within_root(repo, abs_p):
                return False, "blocked_cat_non_repo_path"
            if _is_env_like(abs_p):
                return False, "blocked_cat_env_file"
        return True, None

    return False, "blocked_by_restricted_bash_mode"


@dataclass(frozen=True)
class ToolPolicy:
    repo: Path
    plan_path: Path
    pipeline_path: Path | None
    purpose: str
    bash_mode: str
    unattended: str

    def allow_file_read(self, path: Path) -> tuple[bool, str | None]:
        if _is_env_like(path):
            return False, "reading_env_files_is_blocked"
        if not _within_root(self.repo, path):
            return False, "path_outside_repo"
        return True, None

    def allow_file_write(self, path: Path) -> tuple[bool, str | None]:
        if _is_env_like(path):
            return False, "writing_env_files_is_blocked"
        if not _within_root(self.repo, path):
            return False, "path_outside_repo"

        p = str(self.purpose or "").strip().lower()
        if p == "scaffold_contract":
            pipeline = (self.repo / "pipeline.yml").resolve()
            fsm_dir = (self.repo / ".opencode_fsm").resolve()
            if path == pipeline:
                return True, None
            if _within_root(fsm_dir, path):
                return True, None
            return False, "scaffold_contract_allows_only_pipeline_yml_and_opencode_fsm"
        if p == "repair_contract":
            fsm_dir = (self.repo / ".opencode_fsm").resolve()
            if _within_root(fsm_dir, path):
                return True, None
            return False, "repair_contract_allows_only_opencode_fsm"
        if p.startswith("plan_update") or p in ("mark_done", "block_step"):
            if path == self.plan_path:
                return True, None
            return False, "plan_update_allows_only_plan_md"

        if p == "execute_step":
            if path == self.plan_path:
                return False, "execute_step_disallows_plan_md"
            if self.pipeline_path and path == self.pipeline_path:
                return False, "execute_step_disallows_pipeline_yml"
            return True, None

        if p.startswith("fix_or_replan"):
            if self.pipeline_path and path == self.pipeline_path:
                return False, "fix_or_replan_disallows_pipeline_yml"
            return True, None

        return True, None

    def allow_bash(self, cmd: str) -> tuple[bool, str | None]:
        cmd = cmd.strip()
        if not cmd:
            return False, "empty_command"

        if str(self.bash_mode or "restricted").strip().lower() != "full":
            return _restricted_bash_allowed(cmd, repo=self.repo)

        allowed, reason = cmd_allowed(cmd, pipeline=None)
        if not allowed:
            return False, reason or "blocked"
        if str(self.unattended or "").strip().lower() == "strict" and looks_interactive(cmd):
            return False, "likely_interactive_command_disallowed_in_strict_mode"
        return True, None


def execute_tool_calls(
    calls: Iterable[ToolCall],
    *,
    repo: Path,
    policy: ToolPolicy,
) -> list[ToolResult]:
    results: list[ToolResult] = []

    for call in calls:
        if call.kind == "file":
            data = call.payload if isinstance(call.payload, dict) else {}
            file_path_raw = _xml_unescape(str(data.get("filePath") or "")).strip()
            content = data.get("content")

            if not file_path_raw:
                results.append(ToolResult(kind="file", ok=False, detail={"error": "missing_filePath"}))
                continue

            file_path = Path(file_path_raw).expanduser()
            if not file_path.is_absolute():
                file_path = (repo / file_path).resolve()
            else:
                file_path = file_path.resolve()

            if content is None and ("oldString" in data or "newString" in data):
                old = data.get("oldString")
                new = data.get("newString")
                if isinstance(old, str) and old:
                    old = _xml_unescape(old)
                if isinstance(new, str) and new:
                    new = _xml_unescape(new)
                if new is None:
                    new_s = ""
                elif isinstance(new, str):
                    new_s = new
                else:
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "invalid_newString"})
                    )
                    continue

                if old is None or (isinstance(old, str) and old == ""):
                    ok, reason = policy.allow_file_write(file_path)
                    if not ok:
                        results.append(
                            ToolResult(kind="edit", ok=False,
                                       detail={"filePath": str(file_path), "error": reason or "blocked"})
                        )
                        continue
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(new_s, encoding="utf-8", errors="replace")
                    except Exception as e:
                        results.append(
                            ToolResult(kind="edit", ok=False, detail={
                                "filePath": str(file_path), "error": "write_failed",
                                "exception": type(e).__name__, "message": str(e)[:200],
                            })
                        )
                        continue
                    results.append(
                        ToolResult(kind="edit", ok=True,
                                   detail={"filePath": str(file_path), "bytes": len(new_s), "mode": "replace"})
                    )
                    continue

                if not isinstance(old, str):
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "invalid_oldString"})
                    )
                    continue

                ok, reason = policy.allow_file_write(file_path)
                if not ok:
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": reason or "blocked"})
                    )
                    continue
                if not file_path.exists():
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "not_found"})
                    )
                    continue
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    results.append(
                        ToolResult(kind="edit", ok=False, detail={
                            "filePath": str(file_path), "error": "read_failed",
                            "exception": type(e).__name__, "message": str(e)[:200],
                        })
                    )
                    continue
                matches = raw.count(old)
                if matches <= 0:
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "oldString_not_found"})
                    )
                    continue
                if matches != 1:
                    results.append(
                        ToolResult(kind="edit", ok=False,
                                   detail={"filePath": str(file_path), "error": "oldString_not_unique", "matches": matches})
                    )
                    continue
                updated = raw.replace(old, new_s, 1)
                try:
                    file_path.write_text(updated, encoding="utf-8", errors="replace")
                except Exception as e:
                    results.append(
                        ToolResult(kind="edit", ok=False, detail={
                            "filePath": str(file_path), "error": "write_failed",
                            "exception": type(e).__name__, "message": str(e)[:200],
                        })
                    )
                    continue
                results.append(
                    ToolResult(kind="edit", ok=True,
                               detail={"filePath": str(file_path), "bytes": len(updated), "mode": "replace_once"})
                )
                continue

            if content is None:
                ok, reason = policy.allow_file_read(file_path)
                if not ok:
                    results.append(
                        ToolResult(kind="read", ok=False,
                                   detail={"filePath": str(file_path), "error": reason or "blocked"})
                    )
                    continue
                if not file_path.exists():
                    results.append(
                        ToolResult(kind="read", ok=False,
                                   detail={"filePath": str(file_path), "error": "not_found"})
                    )
                    continue
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    results.append(
                        ToolResult(kind="read", ok=False, detail={
                            "filePath": str(file_path), "error": "read_failed",
                            "exception": type(e).__name__, "message": str(e)[:200],
                        })
                    )
                    continue
                results.append(
                    ToolResult(kind="read", ok=True,
                               detail={"filePath": str(file_path), "content": tail(raw, 20000)})
                )
                continue

            if not isinstance(content, str):
                results.append(
                    ToolResult(kind="write", ok=False,
                               detail={"filePath": str(file_path), "error": "invalid_content"})
                )
                continue
            content = _xml_unescape(content)

            ok, reason = policy.allow_file_write(file_path)
            if not ok:
                results.append(
                    ToolResult(kind="write", ok=False,
                               detail={"filePath": str(file_path), "error": reason or "blocked"})
                )
                continue

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8", errors="replace")
            except Exception as e:
                results.append(
                    ToolResult(kind="write", ok=False, detail={
                        "filePath": str(file_path), "error": "write_failed",
                        "exception": type(e).__name__, "message": str(e)[:200],
                    })
                )
                continue
            results.append(
                ToolResult(kind="write", ok=True,
                           detail={"filePath": str(file_path), "bytes": len(content)})
            )
            continue

        if call.kind == "bash":
            data = call.payload if isinstance(call.payload, dict) else {}
            cmd = _xml_unescape(str(data.get("command") or "")).strip()
            ok, reason = policy.allow_bash(cmd)
            if not ok:
                results.append(ToolResult(kind="bash", ok=False, detail={"command": cmd, "error": reason or "blocked"}))
                continue

            env = _sanitized_env(unattended=str(policy.unattended or "strict"))
            res = run_cmd_capture(cmd, repo, timeout_seconds=60, env=env, interactive=False)
            results.append(
                ToolResult(
                    kind="bash",
                    ok=(res.rc == 0),
                    detail={
                        "command": cmd, "rc": res.rc, "timed_out": res.timed_out,
                        "stdout": tail(res.stdout or "", STDIO_TAIL_CHARS),
                        "stderr": tail(res.stderr or "", STDIO_TAIL_CHARS),
                    },
                )
            )
            continue

        results.append(ToolResult(kind=str(call.kind), ok=False, detail={"error": "unsupported_tool"}))

    return results
