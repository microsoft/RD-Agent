from __future__ import annotations

import re
from pathlib import Path

def audit_eval_script_for_hardcoded_nonzero_score(repo: Path) -> str | None:
    """Best-effort heuristic to catch obvious fake metrics.

    We cannot fully prove "real benchmark execution", but we can block a common
    failure mode: writing a constant non-zero score into metrics.json.
    """
    p = (Path(repo).resolve() / ".opencode_fsm" / "stages" / "evaluation.sh").resolve()
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"failed_to_read_evaluation_sh: {e}"

    # Matches non-zero numbers like 0.92, 1, 2.5, etc. (but not 0 / 0.0 / 0.00).
    nonzero = r"(?:0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(?:\.[0-9]+)?)"
    patterns = [
        re.compile(rf"\bSCORE\s*=\s*['\"]?{nonzero}['\"]?\b", re.IGNORECASE),
        re.compile(rf"\bscore\s*=\s*{nonzero}\b", re.IGNORECASE),
        re.compile(rf"[\"']score[\"']\s*:\s*{nonzero}\b", re.IGNORECASE),
    ]
    bad: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        l = line.strip()
        if not l or l.startswith("#"):
            continue
        if any(pat.search(line) for pat in patterns):
            bad.append(f"{i}: {l}")
        if len(bad) >= 30:
            break
    if not bad:
        return None
    return "hardcoded_nonzero_score_in_.opencode_fsm/stages/evaluation.sh:\n" + "\n".join(bad)

def audit_eval_script_has_real_execution(repo: Path, *, extra_markers: list[str] | None = None) -> str | None:
    """Heuristic: evaluation.sh should *run* something beyond writing JSON.

    The check is benchmark-agnostic and only looks for high-signal "execution markers"
    or doc-derived anchors.
    """
    p = (Path(repo).resolve() / ".opencode_fsm" / "stages" / "evaluation.sh").resolve()
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    markers = [
        "pytest",
        "make ",
        "npm ",
        "node ",
        "docker",
        "inspect ",
    ]
    markers.extend([m for m in (extra_markers or []) if str(m).strip()])
    non_exec_prefixes = (
        "echo",
        "cat",
        "printf",
        "grep",
        "rg",
        "sed",
        "awk",
        "jq",
        "head",
        "tail",
        "test",
        "[",
        "mkdir",
        "date",
        "true",
        "false",
        "cp",
        "mv",
        "rm",
        "touch",
    )
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        first = low.split(maxsplit=1)[0] if low.split() else ""
        if first in non_exec_prefixes:
            continue
        if re.match(r"^[a-z_][a-z0-9_]*=", first):
            continue
        s = low.replace('"', "").replace("'", "").strip()
        if s:
            # Accept common shell exec prefix (e.g., `exec "$PY" -m ...`).
            if s.startswith("exec "):
                s = s[len("exec ") :].strip()
            # Common heredoc pattern for embedded python:
            #   "$PY" - <<'PY'
            #   python3 - <<PY
            if "<<" in s:
                if "<<" in s and ("<<py" in s or "<<-py" in s or "<<\tpy" in s or "<< 'py'" in s or "<<'py'" in s):
                    if "python" in s or "$py" in s or "$opencode_fsm_python" in s:
                        return None
            # Accept `$OPENCODE_FSM_PYTHON` and also wrapper vars like `$PYTHON`.
            if s.startswith("$") or s.startswith("python3") or s.startswith("python"):
                if " -m " in s:
                    return None
                if " -c " in s or s.endswith(" -c"):
                    return None
                if ".py" in s:
                    return None
        if any(m in low for m in markers):
            return None
    return (
        "evaluation.sh does not appear to run any benchmark/evaluation command "
        "(no exec markers, python -m/-c/*.py invocations, or doc-derived anchors found)"
    )

def audit_eval_script_mentions_any_anchor(repo: Path, anchors: list[str]) -> str | None:
    """If we have doc-derived anchors, require evaluation.sh to reference at least one."""
    anchors2 = [str(a).strip().lower() for a in (anchors or []) if str(a).strip()]
    if not anchors2:
        return None
    p = (Path(repo).resolve() / ".opencode_fsm" / "stages" / "evaluation.sh").resolve()
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    # Accept the generic helper-based implementation: anchors are enforced via hints_used.json validation.
    low_all = text.lower()
    if "runner.generic_evaluation" in low_all or "runner.hints_exec" in low_all:
        return None
    # Also accept the script-path invocation form which avoids `runner` module name
    # collisions with target repos that contain their own `runner/` package.
    if "opencode_fsm_runner_root" in low_all and "generic_evaluation.py" in low_all:
        return None

    non_exec_prefixes = (
        "echo",
        "cat",
        "printf",
        "grep",
        "rg",
        "sed",
        "awk",
        "jq",
        "head",
        "tail",
        "test",
        "[",
        "mkdir",
        "date",
        "true",
        "false",
        "cp",
        "mv",
        "rm",
        "touch",
        "export",
    )
    for raw in text.splitlines():
        line = raw.strip().lower()
        if not line or line.startswith("#"):
            continue
        first = line.split(maxsplit=1)[0] if line.split() else ""
        if first in non_exec_prefixes:
            continue
        if re.match(r"^[a-z_][a-z0-9_]*=", first):
            continue
        if any(a in line for a in anchors2):
            return None
    return "evaluation.sh does not appear to use any doc-derived benchmark/eval command hint anchors (likely proxy evaluation)"
