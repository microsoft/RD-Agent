"""
Benchmark dataset configuration and data preparation adaptor for finetune benchmarks.

This module centralizes:
- Mapping of benchmark names to OpenCompass dataset config import paths.
- Optional dataset download / preparation hooks for benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from rdagent.scenarios.finetune.benchmark.data import financeiq_ppl

DownloadFunc = Callable[[], None]


@dataclass
class BenchmarkConfig:
    """
    Configuration for a single benchmark.

    Attributes:
        dataset: Import path for the dataset config in OpenCompass.
        download: Optional function to ensure the dataset is available (e.g. download from HF).
    """

    dataset: str
    download: Optional[DownloadFunc] = None


# Mapping from benchmark_name -> benchmark configuration.
BENCHMARK_CONFIG_DICT: Dict[str, BenchmarkConfig] = {
    # Math Reasoning Benchmarks
    "aime24": BenchmarkConfig(
        dataset="opencompass.configs.datasets.aime2024.aime2024_gen_17d799",
    ),
    "aime25": BenchmarkConfig(
        dataset="opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f",
    ),
    "math": BenchmarkConfig(
        dataset="opencompass.configs.datasets.math.math_0shot_gen_393424",
    ),
    # General Knowledge Benchmarks
    "mmlu": BenchmarkConfig(
        dataset="opencompass.configs.datasets.mmlu.mmlu_gen",
    ),
    # Code Generation Benchmarks
    "humaneval": BenchmarkConfig(
        dataset="opencompass.configs.datasets.humaneval.humaneval_gen",
    ),
    "mbpp": BenchmarkConfig(
        dataset="opencompass.configs.datasets.mbpp.mbpp_gen",
    ),
    # PANORAMA - Patent Analysis Benchmarks (zero-shot)
    "panorama": BenchmarkConfig(
        dataset="opencompass.configs.datasets.panorama.panorama_gen",
    ),
    "panorama_par4pc": BenchmarkConfig(
        dataset="opencompass.configs.datasets.panorama.panorama_par4pc_gen",
    ),
    "panorama_pi4pc": BenchmarkConfig(
        dataset="opencompass.configs.datasets.panorama.panorama_pi4pc_gen",
    ),
    "panorama_noc4pc": BenchmarkConfig(
        dataset="opencompass.configs.datasets.panorama.panorama_noc4pc_gen",
    ),
    # PANORAMA - Patent Analysis Benchmarks (CoT)
    "panorama_par4pc_cot": BenchmarkConfig(
        dataset="opencompass.configs.datasets.panorama.panorama_par4pc_cot_gen",
    ),
    "panorama_pi4pc_cot": BenchmarkConfig(
        dataset="opencompass.configs.datasets.panorama.panorama_pi4pc_cot_gen",
    ),
    "panorama_noc4pc_cot": BenchmarkConfig(
        dataset="opencompass.configs.datasets.panorama.panorama_noc4pc_cot_gen",
    ),
    # ChemCoTBench - Chemistry Reasoning Benchmarks
    "chemcotbench": BenchmarkConfig(
        dataset="opencompass.configs.datasets.chemcotbench.chemcotbench_gen",
    ),
    "chemcotbench_mol_und": BenchmarkConfig(
        dataset="opencompass.configs.datasets.chemcotbench.chemcotbench_mol_und_gen",
    ),
    "chemcotbench_mol_edit": BenchmarkConfig(
        dataset="opencompass.configs.datasets.chemcotbench.chemcotbench_mol_edit_gen",
    ),
    "chemcotbench_mol_opt": BenchmarkConfig(
        dataset="opencompass.configs.datasets.chemcotbench.chemcotbench_mol_opt_gen",
    ),
    "chemcotbench_reaction": BenchmarkConfig(
        dataset="opencompass.configs.datasets.chemcotbench.chemcotbench_reaction_gen",
    ),
    # TableBench - Table Question Answering Benchmarks
    "tablebench_data_analysis": BenchmarkConfig(
        dataset="opencompass.configs.datasets.tablebench.tablebench_data_analysis_gen",
    ),
    "tablebench_fact_checking": BenchmarkConfig(
        dataset="opencompass.configs.datasets.tablebench.tablebench_fact_checking_gen",
    ),
    "tablebench_numerical_reasoning": BenchmarkConfig(
        dataset="opencompass.configs.datasets.tablebench.tablebench_numerical_reasoning_gen",
    ),
    "tablebench_visualization": BenchmarkConfig(
        dataset="opencompass.configs.datasets.tablebench.tablebench_visualization_gen",
    ),
    "tablebench_gen": BenchmarkConfig(
        dataset="opencompass.configs.datasets.tablebench.tablebench_gen",
    ),
    # BioProBench
    "bioprobench_gen": BenchmarkConfig(
        dataset="opencompass.configs.datasets.bioprobench.bioprobench_gen",
    ),
    "bioprobench_ord": BenchmarkConfig(
        dataset="opencompass.configs.datasets.bioprobench.bioprobench_ord",
    ),
    "bioprobench_err": BenchmarkConfig(
        dataset="opencompass.configs.datasets.bioprobench.bioprobench_err",
    ),
    "bioprobench_pqa": BenchmarkConfig(
        dataset="opencompass.configs.datasets.bioprobench.bioprobench_pqa",
    ),
    # Native OpenCompass benchmarks
    "FinanceIQ_ppl": BenchmarkConfig(
        dataset="opencompass.configs.datasets.FinanceIQ.FinanceIQ_gen_e0e6b5",
        download=financeiq_ppl.download_financeiq_dataset,
    ),
}
