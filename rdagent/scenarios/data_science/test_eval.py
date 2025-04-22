from abc import abstractmethod
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.experiment import FBWorkspace
from pathlib import Path


class NoTestEvalError(Exception):
    """Test evaluation is not provided"""


class TestEvalBase:
    """Evaluate a workspace on Test Dataset"""

    @abstractmethod
    def eval(self, competition: str, workspace: FBWorkspace) -> str:
        """eval the workspace as competition, and return the final evaluation result"""


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
        workspace.inject_files(**{"test.csv": (eval_path / "test.csv").read_text()})
        workspace.execute(
            env=self.env,
            entry=f"python grade.py {competition} | tee mle_score.txt",
        )
        workspace.inject_files(**{file: workspace.DEL_KEY for file in ["grade.py", "test.csv"]})
        workspace.execute(env=self.env, entry="chmod 777 mle_score.txt")
        return (workspace.workspace_path / "mle_score.txt").read_text()


class MLETestEval(TestEvalBase):
    """Evaluation for test data for MLE-Bench competition"""

    def __init__(self) -> None:
        super().__init__()
        self.env = get_ds_env(conf_type="mlebench",
                              extra_volumes={f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data"})
        self.env.prepare()

    def eval(self, competition: str, workspace: FBWorkspace) -> str:
        workspace.execute(
            env=self.env,
            entry=f"mlebench grade-sample submission.csv {competition} --data-dir /mle/data | tee mle_score.txt",
        )
        workspace.execute(env=self.env, entry="chmod 777 mle_score.txt")
        return (workspace.workspace_path / "mle_score.txt").read_text()


def get_test_eval() -> TestEvalBase:
    """Get the test evaluation instance"""
    if DS_RD_SETTING.if_using_mle_data:
        return MLETestEval()
    return TestEval()
