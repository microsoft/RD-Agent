#!/usr/bin/env python3
"""自动下载 benchmark 数据集并转换为统一 JSONL 格式。

用法:
    python benchmarks/download.py              # 下载所有 benchmark
    python benchmarks/download.py humaneval    # 只下载指定的
    python benchmarks/download.py --force      # 强制重新下载（覆盖已有数据）
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

BENCHMARKS_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 各数据集的转换器：HuggingFace row → 统一 JSONL 格式
# ---------------------------------------------------------------------------

def _convert_gsm8k(row: dict) -> dict:
    """openai/gsm8k → {"question", "answer"}"""
    return {
        "question": row["question"],
        "answer": row["answer"],
    }


def _convert_humaneval(row: dict) -> dict:
    """openai/openai_humaneval → {"question", "answer", "task_id", "entry_point", "test"}"""
    return {
        "question": row["prompt"],
        "answer": row["canonical_solution"],
        "task_id": row["task_id"],
        "entry_point": row["entry_point"],
        "test": row["test"],
    }


def _convert_mbpp(row: dict) -> dict:
    """google-research-datasets/mbpp → {"question", "answer", "task_id", "test_list"}"""
    return {
        "question": row["text"],
        "answer": row["code"],
        "task_id": row["task_id"],
        "test_list": row["test_list"],
    }


# benchmark 名 → 转换函数
CONVERTERS = {
    "gsm8k": _convert_gsm8k,
    "humaneval": _convert_humaneval,
    "mbpp": _convert_mbpp,
}


def download_benchmark(bench_dir: Path, force: bool = False) -> bool:
    """下载单个 benchmark 的数据。返回 True 如果成功。"""
    config_path = bench_dir / "config.yaml"
    if not config_path.exists():
        return False

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = config.get("name", bench_dir.name)
    source = config.get("source")
    if not source:
        print(f"  [{name}] 跳过：config.yaml 中没有 source 字段")
        return False

    data_dir = bench_dir / "data"
    train_jsonl = data_dir / "train.jsonl"

    if train_jsonl.exists() and not force:
        lines = sum(1 for _ in train_jsonl.open())
        print(f"  [{name}] 已存在 ({lines} 条)，跳过。使用 --force 强制重新下载")
        return True

    hf_dataset = source["hf_dataset"]
    hf_subset = source.get("hf_subset")
    hf_split = source.get("hf_split", "train")
    max_samples = source.get("max_samples", 0)

    converter = CONVERTERS.get(name)
    if converter is None:
        print(f"  [{name}] 跳过：没有对应的转换器（需要在 download.py 中添加）")
        return False

    print(f"  [{name}] 下载 {hf_dataset} (split={hf_split}) ...", end="", flush=True)

    try:
        from datasets import load_dataset
        kwargs = {"split": hf_split}
        if hf_subset:
            ds = load_dataset(hf_dataset, hf_subset, **kwargs)
        else:
            ds = load_dataset(hf_dataset, **kwargs)
    except Exception as e:
        print(f" 失败: {e}")
        return False

    if max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    data_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for row in ds:
            try:
                converted = converter(row)
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"\n    WARNING: 跳过一条数据: {e}")

    print(f" 完成 ({count} 条)")
    return True


def main():
    parser = argparse.ArgumentParser(description="下载 benchmark 数据集")
    parser.add_argument("benchmarks", nargs="*", help="指定要下载的 benchmark（默认全部）")
    parser.add_argument("--force", action="store_true", help="强制重新下载（覆盖已有数据）")
    parser.add_argument("--list", action="store_true", help="列出所有可用 benchmark")
    args = parser.parse_args()

    # 发现所有 benchmark
    all_benchmarks = {}
    for cfg in sorted(BENCHMARKS_ROOT.glob("*/config.yaml")):
        d = cfg.parent
        if d.name.startswith("_"):
            continue
        try:
            c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            all_benchmarks[c.get("name", d.name)] = d
        except Exception:
            continue

    if args.list:
        print(f"可用 benchmark ({len(all_benchmarks)}):")
        for name, d in all_benchmarks.items():
            c = yaml.safe_load((d / "config.yaml").read_text())
            has_data = "✓" if (d / "data" / "train.jsonl").exists() else "✗"
            src = c.get("source", {}).get("hf_dataset", "无")
            print(f"  {has_data} {name:<15s} [{c.get('task_type', '?')}]  source: {src}")
        return

    # 确定要下载的
    if args.benchmarks:
        targets = {}
        for b in args.benchmarks:
            if b not in all_benchmarks:
                print(f"ERROR: 未知 benchmark: {b}")
                print(f"可用: {', '.join(all_benchmarks.keys())}")
                sys.exit(1)
            targets[b] = all_benchmarks[b]
    else:
        targets = all_benchmarks

    print(f"下载 {len(targets)} 个 benchmark:")
    success = 0
    for name, bench_dir in targets.items():
        if download_benchmark(bench_dir, force=args.force):
            success += 1

    print(f"\n完成: {success}/{len(targets)} 成功")


if __name__ == "__main__":
    main()
