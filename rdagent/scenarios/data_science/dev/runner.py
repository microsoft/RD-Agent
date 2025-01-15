import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.developer import Developer
from rdagent.core.exception import RunnerError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.utils.env import DockerEnv, DSDockerConf


class DSRunner(Developer[DSExperiment]):
    def develop(self, exp: DSExperiment) -> DSExperiment:
        ds_docker_conf = DSDockerConf()
        ds_docker_conf.extra_volumes = {f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}": "/kaggle/input"}
        ds_docker_conf.running_timeout_period = 60 * 60  # 1 hours

        de = DockerEnv(conf=ds_docker_conf)

        # execute workflow
        stdout = exp.experiment_workspace.execute(env=de, entry="python main.py")

        score_fp = exp.experiment_workspace.workspace_path / "scores.csv"
        if not score_fp.exists():
            logger.error("Metrics file (scores.csv) is not generated.")
            raise RunnerError(f"Metrics file (scores.csv) is not generated, log is:\n{stdout}")

        submission_fp = exp.experiment_workspace.workspace_path / "submission.csv"
        if not submission_fp.exists():
            logger.error("Submission file (submission.csv) is not generated.")
            raise RunnerError(f"Submission file (submission.csv) is not generated, log is:\n{stdout}")

        exp.result = pd.read_csv(score_fp, index_col=0)
        return exp
