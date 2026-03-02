"""
TableBench 独立测试脚本
运行 TableBench 系列基准测试
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# 1. 设置环境变量（必须在导入 rdagent 之前）
_project_root = Path(__file__).resolve().parents[2]
os.environ["FT_file_path"] = str(_project_root / "git_ignore_folder" / "finetune_files")

import pandas as pd

from rdagent.components.coder.finetune.conf import get_benchmark_env
from rdagent.scenarios.finetune.benchmark.data.adaptor import BENCHMARK_CONFIG_DICT
from rdagent.scenarios.finetune.benchmark.data.default import extract_error_samples
from rdagent.utils.agent.tpl import T


def run_benchmark_simple(
    workspace_path: str,
    model_path_in_docker: str,
    benchmark_name: str,
    gpu_count: int = 4,
    limit: int = 3,
    offset: int = 0,
    max_error_samples: int = 5,
    result_subdir: str = "",
):
    """
    简化的 benchmark 运行器

    Args:
        workspace_path: 本地工作区路径（结果保存位置）
        model_path_in_docker: Docker 内的模型路径
        benchmark_name: benchmark 名称
        gpu_count: GPU 数量
        limit: 样本限制（用于快速测试）
        offset: 数据集采样起始偏移量 (默认: 0)
        max_error_samples: 提取的错误样本数
        result_subdir: 结果子目录 (如 "validation", "test")
    """
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    # 获取 benchmark 配置
    cfg = BENCHMARK_CONFIG_DICT[benchmark_name]

    # 自动下载依赖数据
    if cfg.download is not None:
        cfg.download()

    # 计算 tensor_parallel_size（向下取最接近的 2 的幂）
    tp_size = 1
    power = 0
    while (1 << (power + 1)) <= gpu_count:
        power += 1
    tp_size = 1 << power

    # 生成 OpenCompass 配置文件
    config_content = T("rdagent.scenarios.finetune.benchmark.configs.opencompass_template:template").r(
        model_abbr=f"test-{benchmark_name}",
        model_path=model_path_in_docker,
        is_lora=False,
        lora_path="",
        dataset_imports=[cfg.dataset],
        limit=limit,
        offset=offset,
        num_runs=1,
        pass_k=None,
        work_dir="/workspace",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        max_seq_len=32768,
        max_out_len=8192,
        batch_size=16,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        enable_thinking=False,
    )

    config_file = workspace / "config.py"
    config_file.write_text(config_content)

    # 获取 Docker 环境（启用缓存）
    env = get_benchmark_env()
    env.conf.enable_cache = True

    # 环境变量（用于需要 LLM judge 的 benchmark）
    env_vars = {
        "OC_JUDGE_MODEL": "gpt-5.1",
        "OC_JUDGE_API_KEY": "sk-1234",
        "OC_JUDGE_API_BASE": "http://localhost:3000",
        "OC_JUDGE_RETRY": "3",
    }

    # 在 Docker 中运行 OpenCompass
    if result_subdir:
        benchmark_work_dir = f"/workspace/benchmark_results/{result_subdir}"
    else:
        benchmark_work_dir = "/workspace/benchmark_results"
    cmd = f"opencompass /workspace/config.py --work-dir {benchmark_work_dir}"
    print(f"Running in Docker: {cmd}")
    if offset:
        print(f"Dataset range: [{offset}:{offset + limit}]")

    result = env.run(
        entry=cmd,
        local_path=str(workspace),
        env=env_vars,
    )

    print(f"Exit code: {result.exit_code}")
    if result.exit_code != 0:
        print(f"Error: {result.stdout[-2000:] if result.stdout else 'No output'}")
        raise RuntimeError(f"Benchmark failed with exit code {result.exit_code}")

    # 从本地工作区提取结果
    work_dir = workspace / "benchmark_results"
    if result_subdir:
        work_dir = work_dir / result_subdir
    timestamped_dirs = sorted(work_dir.glob("202*_*"), reverse=True)
    if not timestamped_dirs:
        raise RuntimeError(f"No results found in {work_dir}")

    result_dir = timestamped_dirs[0]
    csv_files = sorted(result_dir.rglob("summary/*.csv"), reverse=True)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {result_dir}")

    # 解析 CSV 结果
    df = pd.read_csv(csv_files[0])
    score_col = [c for c in df.columns if c not in ["dataset", "version", "metric", "mode"]][0]
    pivoted = df.pivot_table(index="dataset", columns="metric", values=score_col, aggfunc="first").to_dict("index")
    benchmark_results = {ds: {k: v for k, v in metrics.items() if pd.notna(v)} for ds, metrics in pivoted.items()}

    # 提取错误样本
    errors = extract_error_samples(result_dir, max_samples=max_error_samples)

    return {"benchmark_results": benchmark_results, "error_samples": errors}


if __name__ == "__main__":
    # 切换到项目根目录（模板解析需要）
    os.chdir(_project_root)

    # ========== 配置区域 ==========
    MODEL = "Qwen/Qwen2.5-1.5B"  # 修改为你的模型名称
    LIMIT = 10  # 样本数限制（None 表示无限制）
    GPU_COUNT = 4  # 你的 GPU 数量

    # Docker 模型路径（自动挂载在 /finetune/models）
    model_path_in_docker = f"/finetune/models/{MODEL}"

    # 创建测试目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_base = _project_root / "git_ignore_folder" / "test" / timestamp

    print("=" * 60)
    print(f"TABLEBENCH TEST: {MODEL} (limit={LIMIT})")
    print(f"Docker model path: {model_path_in_docker}")
    print(f"Output: {test_base}")
    print("=" * 60)

    results_summary = {}

    # TableBench 基准列表
    BENCHMARKS_TO_TEST = [
        "tablebench_data_analysis",  # 数据分析
        "tablebench_fact_checking",  # 事实检查
        "tablebench_numerical_reasoning",  # 数值推理
        "tablebench_visualization",  # 可视化
        # "tablebench_gen",               # 综合（包含上述所有类型）
    ]

    # 运行每个 benchmark
    for benchmark_name in BENCHMARKS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"Running: {benchmark_name}")
        print("=" * 60)

        workspace = test_base / benchmark_name
        result = run_benchmark_simple(
            workspace_path=str(workspace),
            model_path_in_docker=model_path_in_docker,
            benchmark_name=benchmark_name,
            gpu_count=GPU_COUNT,
            limit=LIMIT,
            max_error_samples=5,
        )

        error_samples = result.get("error_samples", [])
        benchmark_results = result.get("benchmark_results", {})

        print(f"  Results: {benchmark_results}")
        print(f"  Error samples: {len(error_samples)}")
        if error_samples:
            print(f"  First error: {error_samples[0]}")

        results_summary[benchmark_name] = {
            "error_count": len(error_samples),
            "benchmark_results": benchmark_results,
        }

    # 打印汇总
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, info in results_summary.items():
        results = info["benchmark_results"]
        print(f"\n{name}:")
        print(f"  Error count: {info['error_count']}")
        for dataset, metrics in results.items():
            print(f"  {dataset}: {metrics}")
