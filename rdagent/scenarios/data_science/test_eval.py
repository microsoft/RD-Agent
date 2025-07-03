from abc import abstractmethod
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.experiment import FBWorkspace


class NoTestEvalError(Exception):
    """Test evaluation is not provided"""


class TestEvalBase:
    """Evaluate a workspace on Test Dataset"""

    @abstractmethod
    def eval(self, competition: str, workspace: FBWorkspace) -> str:
        """eval the workspace as competition, and return the final evaluation result"""

    @abstractmethod
    def valid(self, competition: str, workspace: FBWorkspace) -> tuple[str, int]:
        """eval the workspace as competition, and return the final format check result"""

    @abstractmethod
    def enabled(self, competition) -> bool:
        """able to eval or not"""

    @abstractmethod
    def is_sub_enabled(self, competition: str) -> bool:
        """
        Is subsmiossion file enabled

        If a file like <sample submission csv> is provided; then we think inference from test data to submission file is enabled.
        According test will be enabled as well.

        Why do not we merge `is_sub_enabled` and `enabled`, cases:
        1. The dataset provide evaluation.  But we don't provide submission sample(llm will decide by himself)
        2. We proivde a sample submission. But we don't proivde strict evaluation.

        """
        input_dir = Path(f"{DS_RD_SETTING.local_data_path}/{competition}")
        sample_submission_files = list(input_dir.glob("*sample_submission*.csv")) + list(
            input_dir.glob("*sampleSubmission*.csv")
        )
        return len(sample_submission_files) > 0


class TestEval(TestEvalBase):
    """The most basic version of evaluation for test data"""

    def __init__(self) -> None:
        super().__init__()
        self.env = get_ds_env()

    def eval(self, competition: str, workspace: FBWorkspace) -> str:
        eval_path = Path(f"{DS_RD_SETTING.local_data_path}/{DS_RD_SETTING.eval_sub_dir}/{competition}")
        if not eval_path.exists():
            err_msg = f"No Test Eval provided due to: {eval_path} not found"
            raise NoTestEvalError(err_msg)
        workspace.inject_files(**{"grade.py": (eval_path / "grade.py").read_text()})
        workspace.inject_files(**{"submission_test.csv": (eval_path / "submission_test.csv").read_text()})
        workspace.execute(
            env=self.env,
            entry=f"python grade.py {competition} | tee mle_score.txt",
        )
        workspace.inject_files(**{file: workspace.DEL_KEY for file in ["grade.py", "submission_test.csv"]})
        workspace.execute(env=self.env, entry="chmod 777 mle_score.txt")
        return (workspace.workspace_path / "mle_score.txt").read_text()

    def valid(self, competition: str, workspace: FBWorkspace) -> tuple[str, int]:
        eval_path = Path(f"{DS_RD_SETTING.local_data_path}/{DS_RD_SETTING.eval_sub_dir}/{competition}")
        if not eval_path.exists():
            err_msg = f"No Test Eval provided due to: {eval_path} not found"
            raise NoTestEvalError(err_msg)
        workspace.inject_files(**{"submission_format_valid.py": (eval_path / "valid.py").read_text()})
        workspace.inject_files(**{"submission_test.csv": (eval_path / "submission_test.csv").read_text()})
        submission_result = workspace.run(
            env=self.env,
            entry=f"python submission_format_valid.py {competition}",
        )
        workspace.inject_files(
            **{file: workspace.DEL_KEY for file in ["submission_format_valid.py", "submission_test.csv"]}
        )
        workspace.inject_files(**{"test/mle_submission_format_test.output": submission_result.stdout})
        return submission_result.stdout, submission_result.exit_code

    def enabled(self, competition) -> bool:
        return Path(
            f"{DS_RD_SETTING.local_data_path}/{DS_RD_SETTING.eval_sub_dir}/{competition}/submission_test.csv"
        ).exists()


class MLETestEval(TestEvalBase):
    """Evaluation for test data for MLE-Bench competition"""

    def __init__(self) -> None:
        super().__init__()
        self.env = get_ds_env(
            conf_type="mlebench", extra_volumes={f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data"}
        )
        self.env.prepare()

    def eval(self, competition: str, workspace: FBWorkspace) -> str:
        workspace.execute(
            env=self.env,
            entry=f"mlebench grade-sample submission.csv {competition} --data-dir /mle/data 2>&1 | tee mle_score.txt",
            # NOTE: mlebench does not give output to stdout. so 2>&1 is very necessary !!!!!!
        )
        workspace.execute(env=self.env, entry="chmod 777 mle_score.txt")
        return (workspace.workspace_path / "mle_score.txt").read_text()

    def valid(self, competition: str, workspace: FBWorkspace) -> tuple[str, int]:
        mle_check_code = (
            (Path(__file__).absolute().resolve().parent / "eval_tests" / "mle_submission_format_test.txt")
            .read_text()
            .replace("<competition_id>", competition)
        )
        workspace.inject_files(**{"test/mle_submission_format_test.py": mle_check_code})
        submission_result = workspace.run(env=self.env, entry="python test/mle_submission_format_test.py")

        workspace.inject_files(**{"test/mle_submission_format_test.output": submission_result.stdout})
        return submission_result.stdout, submission_result.exit_code

    def enabled(self, competition) -> bool:
        return True


def get_test_eval() -> TestEvalBase:
    """Get the test evaluation instance"""
    if DS_RD_SETTING.if_using_mle_data:
        return MLETestEval()
    return TestEval()
