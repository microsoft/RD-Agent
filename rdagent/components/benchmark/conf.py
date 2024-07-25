from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)
from dataclasses import field
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

DIRNAME = Path(__file__).absolute().resolve().parent


class BenchmarkSettings(BaseSettings):
    class Config:
        env_prefix = "BENCHMARK_"  # Use BENCHMARK_ as prefix for environment variables

    ground_truth_dir: Path = DIRNAME / "ground_truth"

    bench_data_path: Path = DIRNAME / "example.json"

    bench_test_round: int = 10
    bench_test_case_n: Optional[int] = None  # how many test cases to run; If not given, all test cases will be run

    bench_method_cls: str = "rdagent.components.coder.factor_coder.CoSTEER.FactorCoSTEER"
    bench_method_extra_kwargs: dict = field(
        default_factory=dict,
    )  # extra kwargs for the method to be tested except the task list

    bench_result_path: Path = DIRNAME / "result"
