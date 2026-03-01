"""Benchmark 注册表：自动扫描 benchmarks/*/config.yaml 发现并管理所有 benchmark。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_BENCHMARKS_ROOT = Path(__file__).resolve().parent

_cache: dict[str, BenchmarkInfo] | None = None


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    data_dir: Path
    task_type: str
    description: str
    root: Path
    expose_files: tuple[str, ...] = ()

    @property
    def train_jsonl(self) -> Path:
        return self.data_dir / "train.jsonl"


def discover_benchmarks() -> dict[str, BenchmarkInfo]:
    """扫描 benchmarks/ 下所有含 config.yaml 的子目录，返回 {name: BenchmarkInfo}。"""
    global _cache
    if _cache is not None:
        return _cache

    result: dict[str, BenchmarkInfo] = {}
    for cfg_path in sorted(_BENCHMARKS_ROOT.glob("*/config.yaml")):
        bench_dir = cfg_path.parent
        if bench_dir.name.startswith("_"):
            continue
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue

        name = str(raw.get("name") or bench_dir.name).strip()
        data_dir = bench_dir / "data"
        expose_raw = raw.get("expose_files", [])
        expose_files = tuple(str(f) for f in expose_raw) if isinstance(expose_raw, list) else ()
        info = BenchmarkInfo(
            name=name,
            data_dir=data_dir,
            task_type=str(raw.get("task_type", "unknown")).strip(),
            description=str(raw.get("description", "")).strip(),
            root=bench_dir,
            expose_files=expose_files,
        )
        result[name] = info

    _cache = result
    return result


def get_benchmark(name: str) -> BenchmarkInfo:
    """按名称获取 benchmark，找不到则抛出 KeyError。"""
    benchmarks = discover_benchmarks()
    if name not in benchmarks:
        available = ", ".join(sorted(benchmarks.keys())) or "(none)"
        raise KeyError(f"Unknown benchmark: {name!r}. Available: {available}")
    return benchmarks[name]


def list_benchmarks() -> list[str]:
    """返回所有已注册 benchmark 名称（排序）。"""
    return sorted(discover_benchmarks().keys())
