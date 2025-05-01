from pathlib import Path
import re
from typing import Any

import pandas as pd

from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import QTDockerEnv, QlibCondaConf, LocalEnv


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs) -> str:
        # qtde = QTDockerEnv()  # This is for the docker environment
        qtde = LocalEnv(conf=QlibCondaConf())  # This is for the local environment
        qtde.prepare()

        # Run the Qlib backtest
        execute_qlib_log = qtde.run(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env,
        )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        # TODO: We should handle the case when Docker times out.
        execute_log = qtde.run(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env,
        )

        pattern = r"(Epoch\d+: train -?\d+\.\d+, valid -?\d+\.\d+|early stop|best score: -?\d+\.\d+ @ \d+ epoch)"
        matches = re.findall(pattern, execute_qlib_log)
        execute_qlib_log = "\n".join(matches)

        quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            logger.error("No result file found.")
            return None, execute_qlib_log

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            return None, execute_qlib_log
