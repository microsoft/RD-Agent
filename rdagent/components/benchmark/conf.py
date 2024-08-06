from dataclasses import field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv(verbose=True, override=True)


DIRNAME = Path("./")


class BenchmarkSettings(BaseSettings):
    class Config:
        env_prefix = "BENCHMARK_"
        """Use `BENCHMARK_` as prefix for environment variables"""

    ground_truth_dir: Path = DIRNAME / "ground_truth"
    """ground truth dir"""

    bench_data_path: Path = DIRNAME / "example.json"
    """data for benchmark"""

    bench_test_round: int = 10
    """how many rounds to run, each round may cost 10 minutes"""

    bench_test_case_n: Optional[int] = None
    """how many test cases to run; If not given, all test cases will be run"""

    bench_method_cls: str = "rdagent.components.coder.factor_coder.CoSTEER.FactorCoSTEER"
    """method to be used for test cases"""

    bench_method_extra_kwargs: dict = field(
        default_factory=dict,
    )
    """extra kwargs for the method to be tested except the task list"""

    bench_result_path: Path = DIRNAME / "result"
    """result save path"""
