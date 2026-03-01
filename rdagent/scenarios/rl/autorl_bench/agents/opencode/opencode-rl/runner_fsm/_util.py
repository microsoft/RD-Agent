from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# --- General utilities ---

def _is_truthy(value: str | None) -> bool:
    v = str(value or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _parse_json_str_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(str(raw))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for x in data:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if s:
            out.append(s)
    return out

def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data

def _find_hf_test_parquet(repo_root: Path) -> Path | None:
    """Find a Hugging Face dataset test split parquet file (best-effort)."""
    repo_root = Path(repo_root).resolve()
    p0 = (repo_root / "main" / "test-00000-of-00001.parquet").resolve()
    if p0.exists():
        return p0
    cands: list[Path] = []
    for p in repo_root.rglob("test-*.parquet"):
        try:
            if p.is_file():
                cands.append(p.resolve())
        except Exception:
            continue
    cands.sort()
    return cands[0] if cands else None

def _ensure_openai_v1_base(base_url: str) -> str:
    b = str(base_url or "").strip().rstrip("/")
    if not b:
        return ""
    return b if b.endswith("/v1") else b + "/v1"


# --- Path utilities (merged from paths.py) ---

def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except Exception:
        return False

def relpath_or_none(path: Path, base: Path) -> str | None:
    if not is_relative_to(path, base):
        return None
    return str(path.relative_to(base))

def resolve_config_path(repo: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = repo / p
    return p.resolve()

def resolve_workdir(repo: Path, workdir: str | None) -> Path:
    if not workdir or not str(workdir).strip():
        return repo
    p = Path(workdir).expanduser()
    if not p.is_absolute():
        p = repo / p
    p = p.resolve()
    if not is_relative_to(p, repo):
        raise ValueError(f"workdir must be within repo: {p} (repo={repo})")
    return p
