from rdagent.core.developer import Developer
import pandas as pd
from rdagent.core.exception import CoderError
from rdagent.utils.env import DockerEnv, DSDockerConf
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment

class DSRunner(Developer[DSExperiment]):
    def develop(self, exp: DSExperiment) -> DSExperiment:
        ds_docker_conf = DSDockerConf()
        ds_docker_conf.extra_volumes = {f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}": "/kaggle/input"}
        
        de = DockerEnv(conf=ds_docker_conf)
        
        # execute workflow
        exp.experiment_workspace.execute(env=de, entry="python main.py")
        submission_fp = exp.experiment_workspace.workspace_path / "submission.csv"
        score_fp = exp.experiment_workspace.workspace_path / "scores.csv"
        
        if not submission_fp.exists():
            logger.error("Submission file (submission.csv) is not generated.")
            raise CoderError("Submission file (submission.csv) is not generated.")
        
        if not score_fp.exists():
            logger.error("Metrics file (scores.csv) is not generated.")
            raise CoderError("Metrics file (scores.csv) is not generated.")
        
        exp.result = pd.read_csv(score_fp, index_col=0)
        return exp