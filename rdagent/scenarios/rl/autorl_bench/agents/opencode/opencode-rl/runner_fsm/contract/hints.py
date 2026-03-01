from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ContractHints:
    """Best-effort command hints extracted from the repo itself (no hardcoding)."""

    commands: list[str]
    anchors: list[str]

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(?P<body>.*?)\n```", re.DOTALL)

_PROMPT_PREFIX_RE = re.compile(r"^(?P<prefix>(?:\$|>>>|\.\.\.)\s+)")

_COMMANDISH_FIRST = {
    "bash",
    "sh",
    "zsh",
    "python",
    "python3",
    "pip",
    "pip3",
    "uv",
    "poetry",
    "conda",
    "make",
    "pytest",
    "docker",
    "git",
    # Shell builtins are still useful anchors (e.g., `cd repo`).
    "cd",
    "export",
    "env",
    "source",
}

def _extract_workflow_candidates(
    repo: Path,
    *,
    interest_re: re.Pattern[str],
    max_files: int,
    max_candidates: int,
) -> list[str]:
    """Best-effort: extract runnable-ish command scripts from CI workflows.

    Strategy:
    - Parse `.github/workflows/*.yml(yaml)` and collect step `run:` scripts.
    - When an "interesting" evaluation command is found (pytest/eval/benchmark/...),
      also prepend a small prelude of preceding run steps (up to 3). This captures
      common setup (e.g., `docker build` before `docker run ... pytest`).
    """
    repo = Path(repo).resolve()
    out: list[str] = []
    seen: set[str] = set()

    wf_root = (repo / ".github" / "workflows").resolve()
    wf_paths: list[Path] = []
    if wf_root.exists():
        for pat in ("*.yml", "*.yaml"):
            wf_paths.extend(sorted(wf_root.glob(pat)))
        wf_paths = [p for p in wf_paths if p.is_file()]
        wf_paths = wf_paths[: int(max(1, max_files))]
    for p in wf_paths:
        if len(out) >= int(max(1, max_candidates)):
            break
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        scripts: list[str] = []
        try:
            import yaml  # type: ignore
        except Exception:
            continue
        try:
            data = yaml.safe_load(text)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for _job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            steps = job.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                run = step.get("run")
                if not isinstance(run, str):
                    continue
                s = run.strip()
                if s:
                    scripts.append(s)
        if not scripts:
            continue

        step_cmds: list[str] = []
        for s in scripts:
            lines: list[str] = []
            for raw in str(s or "").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(line)
            joined: list[str] = []
            cur = ""
            for line in lines:
                if cur:
                    if cur.rstrip().endswith("\\"):
                        cur = cur.rstrip()[:-1].rstrip() + " " + line.lstrip()
                        continue
                    joined.append(cur)
                    cur = ""
                # Skip orphan option lines.
                if line.startswith("-"):
                    continue
                cur = line
            if cur:
                joined.append(cur)
            if len(joined) >= 2:
                step_cmds.append("\n".join(joined))
            elif joined:
                step_cmds.append(joined[0])

        for idx, cmd in enumerate(step_cmds):
            if len(out) >= int(max(1, max_candidates)):
                break
            if not interest_re.search(cmd):
                continue
            prelude = [c for c in step_cmds[max(0, idx - 3) : idx] if c.strip()]
            combined = ("\n".join(prelude + [cmd])).strip()
            for item in (combined, cmd.strip()):
                if not item or item in seen:
                    continue
                seen.add(item)
                out.append(item)
                if len(out) >= int(max(1, max_candidates)):
                    break

    return out

def _extract_anchors(hints: list[str]) -> list[str]:
    """Extract high-signal tokens we can use to audit "did you use doc hints?".

    This is intentionally heuristic and benchmark-agnostic.
    """
    skip_first = {
        "cd",
        "bash",
        "sh",
        "zsh",
        "import",
        "from",
        "python",
        "python3",
        "pip",
        "pip3",
        "uv",
        "conda",
        "poetry",
        "make",
        "npm",
        "node",
        "docker",
        "git",
        "sudo",
    }
    skip_tokens = {
        "install",
        "uninstall",
        "run",
        "start",
        "stop",
        "setup",
        "build",
        "download",
        "clone",
        "create",
        "remove",
        "update",
        "upgrade",
        "requirements.txt",
        "requirements-ml.txt",
    }
    out: list[str] = []
    seen: set[str] = set()
    for raw in hints or []:
        cmd2 = str(raw or "").strip()
        if not cmd2:
            continue
        if cmd2.startswith("> "):
            # Markdown quote prefix occasionally appears inside fenced blocks.
            cmd2 = cmd2[2:].lstrip()
        m = _PROMPT_PREFIX_RE.match(cmd2)
        if m:
            cmd2 = cmd2[m.end("prefix") :].lstrip()
        try:
            tokens = shlex.split(cmd2)
        except Exception:
            # Fallback tokenization; good enough for anchors.
            tokens = [t for t in re.split(r"\s+", str(cmd2 or "").strip()) if t]
        if not tokens:
            continue
        # Prefer module invocations: `python -m pkg.module ...`
        if "-m" in tokens:
            try:
                mod = tokens[tokens.index("-m") + 1]
            except Exception:
                mod = ""
            mod = str(mod or "").strip()
            if mod and len(mod) >= 6:
                if mod not in seen:
                    seen.add(mod)
                    out.append(mod)
                pkg = mod.split(".", 1)[0].strip()
                if pkg and pkg not in seen and len(pkg) >= 5:
                    seen.add(pkg)
                    out.append(pkg)
                continue

        first = str(tokens[0] or "").strip()
        if not first:
            continue
        if first in skip_first:
            # If the first token is too generic, try to grab a later high-signal token.
            for t in tokens[1:6]:
                tt = str(t or "").strip()
                if not tt or tt.startswith("-"):
                    continue
                if tt.lower() in skip_tokens:
                    continue
                # Prefer dotted modules/binaries or long-ish names (avoid `install`, `run` etc).
                if "." in tt and len(tt) >= 6 and tt not in seen:
                    seen.add(tt)
                    out.append(tt)
                    break
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{5,}", tt) and tt not in seen:
                    seen.add(tt)
                    out.append(tt)
                    break
            continue

        # Use the binary name when it is not generic.
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{4,}", first) and first not in seen:
            seen.add(first)
            out.append(first)
            if "." in first:
                pkg = first.split(".", 1)[0].strip()
                if pkg and pkg not in seen and len(pkg) >= 5:
                    seen.add(pkg)
                    out.append(pkg)
    return out[:12]

def suggest_contract_hints(repo: Path, *, max_files: int = 8, max_candidates: int = 20) -> ContractHints:
    """Extract candidate evaluation/benchmark commands from repo docs.

    - Generic: only uses repo content; no benchmark-specific logic.
    - Best-effort: returns an empty list if nothing is found.
    """
    repo = Path(repo).resolve()
    md_paths: list[Path] = []
    for pat in ("README*.md", "docs/**/*.md"):
        md_paths.extend(sorted(repo.glob(pat)))
    md_paths = [p for p in md_paths if p.is_file()]
    md_paths = md_paths[: int(max(1, max_files))]

    interest_re = re.compile(
        r"(?i)("
        r"\beval\b|"
        r"evaluate|evaluation|"
        r"benchmark|leaderboard|quick\s+start|"
        r"pytest|inspect"
        r")"
    )
    seen: set[str] = set()
    candidates: list[str] = []

    for p in md_paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        text = text[:200_000]
        for m in _FENCE_RE.finditer(text):
            body = (m.group("body") or "").strip()
            if not body:
                continue

            # Reconstruct multi-line shell commands that use `\` line continuations.
            block_cmds: list[str] = []
            cur = ""
            for raw in body.splitlines():
                line = str(raw or "").strip()
                if not line:
                    continue
                if line.startswith("> "):
                    # Markdown quote prefix occasionally appears inside fenced blocks.
                    line = line[2:].lstrip()
                m2 = _PROMPT_PREFIX_RE.match(line)
                if m2:
                    line = line[m2.end("prefix") :].lstrip()
                if not line or line.startswith("#"):
                    continue
                low = line.lower()
                if "| bash" in low or "| sh" in low:
                    continue

                if cur:
                    if cur.rstrip().endswith("\\"):
                        cur = cur.rstrip()[:-1].rstrip() + " " + line.lstrip()
                        continue
                    block_cmds.append(cur)
                    cur = ""

                # Skip orphan option lines.
                if line.startswith("-"):
                    continue
                cur = line

            if cur:
                block_cmds.append(cur)

            for cmd in block_cmds:
                s = str(cmd or "").strip()
                if not s:
                    continue
                if s.startswith("> "):
                    # Markdown quote prefix occasionally appears inside fenced blocks.
                    s = s[2:].lstrip()
                m3 = _PROMPT_PREFIX_RE.match(s)
                if m3:
                    s = s[m3.end("prefix") :].lstrip()
                if not s:
                    continue
                try:
                    parts = shlex.split(s, posix=True)
                except Exception:
                    parts = [t for t in re.split(r"\s+", s) if t]
                if not parts:
                    continue
                first = str(parts[0] or "").strip()
                if not first:
                    continue
                if first.startswith(("@", "*", "[", "(", "{")):
                    continue
                # Reject key/value style lines from BibTeX/YAML/etc (e.g. `title = {...}`).
                if re.match(r"^[A-Za-z][A-Za-z0-9_]{2,}\\s*=\\s*", s):
                    continue
                if not (
                    first.startswith((".", "/"))
                    or "/" in first
                    or first.endswith((".py", ".sh"))
                    or first in _COMMANDISH_FIRST
                    or (
                        re.fullmatch(r"[a-z][a-z0-9_.-]{2,}", first)
                        and any(ch in first for ch in "._-")
                    )
                ):
                    continue
                if not interest_re.search(s):
                    continue
                if s in seen:
                    continue
                seen.add(s)
                candidates.append(s)
                if len(candidates) >= int(max(1, max_candidates)):
                    anchors = _extract_anchors(candidates)
                    return ContractHints(commands=candidates, anchors=anchors)

    # Fall back to CI workflow hints when README/docs are insufficient.
    if len(candidates) < int(max(1, max_candidates)):
        wf_candidates = _extract_workflow_candidates(
            repo,
            interest_re=interest_re,
            max_files=max_files,
            max_candidates=int(max(0, int(max_candidates) - len(candidates))),
        )
        for cmd in wf_candidates:
            if cmd in seen:
                continue
            seen.add(cmd)
            candidates.append(cmd)
            if len(candidates) >= int(max(1, max_candidates)):
                anchors = _extract_anchors(candidates)
                return ContractHints(commands=candidates, anchors=anchors)

    anchors = _extract_anchors(candidates)
    return ContractHints(commands=candidates, anchors=anchors)
