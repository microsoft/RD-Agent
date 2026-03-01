from __future__ import annotations

import re

_SCORE_TOKEN_RE = re.compile(
    r"(?i)\b(?P<key>pass@1\+?|accuracy|score)\b[^0-9%]{0,12}(?P<val>\d+(?:\.\d+)?)(?P<pct>\s*%)?"
)


def _normalize_score(value: float, *, had_percent: bool) -> float | None:
    v = float(value)
    if had_percent:
        v = v / 100.0
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    if v < 0.0 or v > 1.0:
        return None
    return float(v)


def _parse_pytest_counts(text: str) -> tuple[int, int, int] | None:
    """Extract (passed, failed, errors) from pytest output text. Returns None if no counts found."""
    t = re.sub(r"\x1b\[[0-9;]*m", "", str(text or ""))
    passed_ms = list(re.finditer(r"(?i)\b(\d+)\s+passed\b", t))
    failed_ms = list(re.finditer(r"(?i)\b(\d+)\s+failed\b", t))
    errors_ms = list(re.finditer(r"(?i)\b(\d+)\s+error(?:s)?\b", t))
    try:
        passed = int(passed_ms[-1].group(1)) if passed_ms else 0
    except Exception:
        passed = 0
    try:
        failed = int(failed_ms[-1].group(1)) if failed_ms else 0
    except Exception:
        failed = 0
    try:
        errors = int(errors_ms[-1].group(1)) if errors_ms else 0
    except Exception:
        errors = 0
    total = passed + failed + errors
    if total > 0:
        return (passed, failed, errors)
    return None


def _extract_score_from_text(text: str) -> tuple[float | None, str]:
    """Best-effort score extraction from stdout/stderr (generic, benchmark-agnostic)."""
    t = str(text or "")
    t = re.sub(r"\x1b\[[0-9;]*m", "", t)
    last: dict[str, tuple[float, bool]] = {}
    for m in _SCORE_TOKEN_RE.finditer(t):
        key = str(m.group("key") or "").strip().lower()
        raw = str(m.group("val") or "").strip()
        had_pct = bool((m.group("pct") or "").strip())
        try:
            val = float(raw)
        except Exception:
            continue
        last[key] = (val, had_pct)
    for key in ("pass@1", "pass@1+", "accuracy", "score"):
        if key in last:
            val, had_pct = last[key]
            norm = _normalize_score(val, had_percent=had_pct)
            if norm is not None:
                return norm, f"text:{key}"
    return None, "no_score_in_text"


def _extract_score_from_json_obj(obj: object) -> tuple[float | None, str]:
    """Best-effort score extraction from JSON-like objects."""
    pairs: list[tuple[str, float]] = []
    _item = object()
    stack: list[object] = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, tuple) and len(x) == 3 and x[0] is _item:
            _, k, v = x
            kk = str(k or "").strip().lower()
            if isinstance(v, (int, float)):
                pairs.append((kk, float(v)))
            stack.append(v)
            continue
        if isinstance(x, dict):
            for k, v in reversed(list(x.items())):
                stack.append((_item, k, v))
        elif isinstance(x, list):
            for it in reversed(x):
                stack.append(it)
    for needle in ("pass@1", "pass_at_1", "pass@1+", "pass_at_1_plus", "accuracy", "score"):
        for k, v in reversed(pairs):
            if needle in k:
                norm = _normalize_score(v, had_percent=False)
                if norm is not None:
                    return norm, f"json:{needle}"
    return None, "no_score_in_json"
