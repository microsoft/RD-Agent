from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageCache:
    stage: str
    ok: bool
    timestamp: float
    pipeline_hash: str
    ttl_seconds: int
    extra: dict[str, Any] = field(default_factory=dict)


def _cache_dir(repo: Path) -> Path:
    return (repo / ".opencode_fsm" / "cache").resolve()


def _file_hash(path: Path) -> str:
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    return hashlib.sha256(data).hexdigest()


def _pipeline_hash(repo: Path) -> str:
    return _file_hash((repo / "pipeline.yml").resolve())


def _default_ttl() -> int:
    raw = os.environ.get("OPENCODE_FSM_CACHE_TTL", "3600")
    try:
        return max(0, int(str(raw).strip()))
    except Exception:
        return 3600


def _cache_globally_enabled() -> bool:
    raw = os.environ.get("OPENCODE_FSM_CACHE_ENABLED", "1")
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def load_stage_cache(repo: Path, stage_name: str) -> StageCache | None:
    """Load and validate a stage cache entry. Returns None if invalid/missing/expired."""
    if not _cache_globally_enabled():
        return None
    path = _cache_dir(repo) / f"{stage_name}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("ok") is not True:
        return None

    ttl = int(data.get("ttl_seconds", 0) or 0)
    ts = float(data.get("timestamp", 0) or 0)
    if ttl > 0 and (time.time() - ts) > ttl:
        return None

    current_ph = _pipeline_hash(repo)
    if current_ph and data.get("pipeline_hash") != current_ph:
        return None

    extra = data.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}

    if stage_name == "bootstrap":
        bootstrap_hash = extra.get("bootstrap_hash", "")
        if bootstrap_hash:
            current_bh = _file_hash((repo / ".opencode_fsm" / "bootstrap.yml").resolve())
            if current_bh and current_bh != bootstrap_hash:
                return None

    if stage_name == "deploy":
        rp = extra.get("runtime_env_path")
        if isinstance(rp, str) and rp:
            if not Path(rp).exists():
                return None

    if stage_name == "rollout":
        rp = extra.get("rollout_path")
        if isinstance(rp, str) and rp:
            if not Path(rp).exists():
                return None

    return StageCache(
        stage=str(data.get("stage", stage_name)),
        ok=True,
        timestamp=ts,
        pipeline_hash=str(data.get("pipeline_hash", "")),
        ttl_seconds=ttl,
        extra=extra,
    )


def save_stage_cache(repo: Path, stage_name: str, **extra: Any) -> None:
    """Persist a successful stage result to the cache."""
    d = _cache_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    obj = {
        "stage": stage_name,
        "ok": True,
        "timestamp": time.time(),
        "pipeline_hash": _pipeline_hash(repo),
        "ttl_seconds": _default_ttl(),
        "extra": extra,
    }
    try:
        (d / f"{stage_name}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def invalidate_stage_cache(repo: Path, stage_name: str) -> None:
    path = _cache_dir(repo) / f"{stage_name}.json"
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def invalidate_all_caches(repo: Path) -> None:
    d = _cache_dir(repo)
    try:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass
