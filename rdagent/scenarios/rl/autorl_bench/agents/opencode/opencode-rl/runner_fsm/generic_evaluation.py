from __future__ import annotations

import os
import sys

# Support running either as a module (`python -m runner.generic_evaluation`) or as a
# script (`python $OPENCODE_FSM_RUNNER_ROOT/runner/generic_evaluation.py`). The latter
# avoids module-name collisions with target repos that may contain their own `runner/`
# package.
if __package__ in (None, ""):
    _file = os.path.abspath(__file__)
    _SCRIPT_DIR = os.path.dirname(_file)
    _ROOT = os.path.dirname(_SCRIPT_DIR)
    # When executed as a script, Python prepends the script directory (runner/)
    # to sys.path. That can shadow stdlib modules like `types` via runner/types.py.
    # Fix by removing runner/ from sys.path and adding the repo root instead.
    root_s = str(_ROOT)
    script_s = str(_SCRIPT_DIR)
    try:
        while script_s in sys.path:
            sys.path.remove(script_s)
        while root_s in sys.path:
            sys.path.remove(root_s)
    except Exception:
        pass
    sys.path.insert(0, root_s)

import json
import time
from pathlib import Path

if __package__ in (None, ""):
    from runner.hints.executor import run_hints  # type: ignore
    from runner._util import _is_truthy, _read_json_object  # type: ignore
else:
    from .hints import run_hints
    from ._util import _is_truthy, _read_json_object

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def _reward_average_from_rollout(repo_root: Path) -> tuple[bool, float, dict[str, int] | None, str]:
    """Compute score as average `reward` across rollout samples (best-effort)."""
    rollout_path = (repo_root / ".opencode_fsm" / "rollout.json").resolve()
    if not rollout_path.exists():
        return False, 0.0, None, f"missing_rollout_json: {rollout_path}"

    rollout = _read_json_object(rollout_path)
    if rollout is None:
        return False, 0.0, None, "rollout_json_not_object"

    paths = rollout.get("paths")
    if not isinstance(paths, dict):
        return False, 0.0, None, "rollout_json_missing_paths"

    raw = paths.get("samples_jsonl")
    if not isinstance(raw, str) or not raw.strip():
        return False, 0.0, None, "rollout_json_missing_paths.samples_jsonl"

    samples_path = Path(raw.strip())
    if not samples_path.is_absolute():
        samples_path = (repo_root / samples_path).resolve()
    if not samples_path.exists():
        return False, 0.0, None, f"samples_jsonl_not_found: {samples_path}"

    total = 0
    reward_sum = 0.0
    bad = 0
    try:
        with samples_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    bad += 1
                    continue
                if not isinstance(obj, dict):
                    bad += 1
                    continue
                r = obj.get("reward")
                if not isinstance(r, (int, float)):
                    bad += 1
                    continue
                reward_sum += float(r)
                total += 1
    except Exception as e:
        return False, 0.0, None, f"failed_to_read_samples_jsonl: {e}"

    if total <= 0:
        return False, 0.0, {"samples": 0, "bad_lines": bad}, "no_valid_reward_samples"

    score = reward_sum / float(total)
    return True, float(score), {"samples": int(total), "bad_lines": int(bad)}, "reward_average"

def main() -> int:
    repo_root = Path(os.environ.get("OPENCODE_FSM_REPO_ROOT") or ".").resolve()
    metrics_path = Path(os.environ.get("OPENCODE_FSM_METRICS_PATH") or ".opencode_fsm/metrics.json")
    if not metrics_path.is_absolute():
        metrics_path = (repo_root / metrics_path).resolve()

    hints_path = (repo_root / ".opencode_fsm" / "hints_used.json").resolve()
    hints_run_path = (repo_root / ".opencode_fsm" / "hints_run.json").resolve()
    require_hints = _is_truthy(os.environ.get("OPENCODE_FSM_REQUIRE_HINTS"))

    try:
        timeout = int(os.environ.get("OPENCODE_FSM_HINT_TIMEOUT_SECONDS") or 600)
    except Exception:
        timeout = 600
    try:
        max_attempts = int(os.environ.get("OPENCODE_FSM_HINT_MAX_ATTEMPTS") or 3)
    except Exception:
        max_attempts = 3

    res = run_hints(repo=repo_root, max_attempts=max_attempts, timeout_seconds=timeout, env=dict(os.environ))
    try:
        _write_json(hints_run_path, dict(res) if isinstance(res, dict) else {"value": res})
    except Exception:
        pass
    ok = bool(res.get("ok") is True)
    score = float(res.get("score") or 0.0)
    hint_reason = str(res.get("reason") or "")

    hints_used = {
        "ok": bool(ok),
        "used_anchors": list(res.get("used_anchors") or []),
        "commands": [
            str(a.get("sanitized") or a.get("raw") or "").strip()
            for a in (res.get("attempts") or [])
            if isinstance(a, dict) and str(a.get("sanitized") or a.get("raw") or "").strip()
        ],
        "reason": "" if ok else str(res.get("reason") or "hint_command_failed"),
    }
    _write_json(hints_path, hints_used)

    # If there are no runnable hints, fall back to computing score from rollout samples.
    if not ok and hint_reason == "no_hints" and not require_hints:
        ok2, score2, counts2, reason2 = _reward_average_from_rollout(repo_root)
        metrics = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "ok": bool(ok2),
            "score": float(score2) if ok2 else 0.0,
            "reason": str(reason2),
        }
        if isinstance(counts2, dict):
            metrics["counts"] = dict(counts2)
        _write_json(metrics_path, metrics)
        return 0 if ok2 else 1

    metrics = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "ok": bool(ok),
        "score": score if ok else 0.0,
        "reason": hint_reason,
    }
    _write_json(metrics_path, metrics)

    if require_hints and not ok:
        return 2
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
