from dotenv import load_dotenv
load_dotenv(verbose=True, override=True)
from dataclasses import field
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic_settings import BaseSettings

DIRNAME = Path(__file__).absolute().resolve().parent

BENCHMARK_VERSION = Literal["paper", "amcV01", "amcV02train", "amcV02test"]

class BenchmarkSettings(BaseSettings):

    ground_truth_dir: Path = DIRNAME / "ground_truth"

    bench_version: Union[BENCHMARK_VERSION, str] = "paper"

    bench_test_round: int = 20 
    bench_test_case_n: Optional[int] = None  # how many test cases to run; If not given, all test cases will be run

    bench_method_cls: str = "scripts.factor_implementation.baselines.naive.one_shot.OneshotFactorGen"
    bench_method_extra_kwargs: dict = field(
        default_factory=dict,
    )  # extra kwargs for the method to be tested except the task list

    bench_result_path: Path = DIRNAME / "result"