from rdagent.core.task_generator import TaskGenerator
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.utils.env import QTDockerEnv
# from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback
# from rdagent.core.experiment import Experiment

from pathlib import Path
import pandas as pd
import os
import shutil
import pickle
from typing import Optional


DIRNAME = Path(__file__).absolute().resolve().parent
RESULT_DIR = Path("/home/finco/quant_result")
MLRUNS_DIR = Path("/home/finco/RDAgent_MS/RD-Agent/rdagent/scenarios/qlib/task_generator/env_factor/mlruns/1")

class QlibFactorRunnerContext:
    def __init__(self, last_experiment: Optional[pd.DataFrame] = None, sota: Optional[pd.DataFrame] = None, alpha158: Optional[pd.DataFrame] = None):
        self.last_experiment = last_experiment
        self.sota = sota
        self.alpha158 = alpha158

    def update_last_experiment(self, new_result: pd.DataFrame):
        self.last_experiment = new_result

    def update_sota(self, new_sota: pd.DataFrame):
        self.sota = new_sota

    def update_alpha158(self, new_alpha158: pd.DataFrame):
        self.alpha158 = new_alpha158

    def setup(self):
        """
        Run a Docker command to generate initial experiment results, and update last_experiment and alpha158 with those results.
        """
        DATA_PATH = "/home/finco/test_result.pkl"
        with open(DATA_PATH, 'rb') as f:
            exp_res = pickle.load(f)
        self.update_last_experiment(exp_res)
        self.update_alpha158(exp_res)

        # qtde = QTDockerEnv()
        # qtde.prepare()  # Preparing the environment
        # qtde.prepare()
        
        # # Run the Docker command
        # result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="rm -r mlruns", env={"PYTHONPATH": "./"})
        # # Run the Qlib backtest
        # result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="qrun conf.yaml", env={"PYTHONPATH": "./"})
        
        # # Check for new directories in MLRUNS_DIR
        # new_dir = {d.name for d in MLRUNS_DIR.iterdir() if d.is_dir()}
        # new_dir_name = new_dir.pop()
        # pkl_path = MLRUNS_DIR / new_dir_name / 'artifacts/portfolio_analysis/port_analysis_1day.pkl'

        # if not pkl_path.exists():
        #     print(f"File {pkl_path} does not exist.")
        #     return None

        # with open(pkl_path, 'rb') as f:
        #     result = pickle.load(f)

        # # Check if the result is valid and is a DataFrame
        # if isinstance(result, pd.DataFrame):
        #     if not result.empty:
        #         self.update_last_experiment(result)
        #         self.update_alpha158(result)
        #     else:
        #         print("Result DataFrame is empty.")
        # else:
        #     print("Data format error.")

class QlibFactorRunner(TaskGenerator[QlibFactorExperiment]):
    def __init__(self, last_experiment: Optional[pd.DataFrame] = None, sota: Optional[pd.DataFrame] = None, alpha158: Optional[pd.DataFrame] = None):
        self.sota = None
        self.alpha158 = None
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`

    - TODO: implement a qlib handler
    """
    def run_qlib(exp = None):
        # prepare foldre
        # docker run
        # get metrics
        # if exp is None run alpha158
        if exp is None:
            ...
            # store metrics to self.alpha158
            # store metrics to sota also

    def generate(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        # if self.alpha158 is None:
        #     self.run_qlib()
        
        # self.run_qlib(exp)

        
        # Process factor data format
        combined_factors = self.process_factor_data(exp)
        print("Success in processing factor data.")
        # TODO: Call Docker, pass the combined factors to Docker, and generate backtest results
        # result = self.test_docker()
        # print(result)
        DATA_PATH = "/home/finco/test_result2.pkl"
        with open(DATA_PATH, 'rb') as f:
            exp_res = pickle.load(f)
        exp.result = exp_res
        
        return exp

    def process_factor_data(self, exp: QlibFactorExperiment) -> pd.DataFrame:
        """
        Process and combine factor data from experiment implementations.

        Args:
            exp (ASpecificExp): The experiment containing factor data.

        Returns:
            pd.DataFrame: Combined factor data without NaN values.
        """
        factor_dfs = []

        # Iterate over sub-implementations and execute them to get each factor data
        for implementation in exp.sub_implementations:
            message, df = implementation.execute()

            # Check if factor generation was successful
            if 'Execution succeeded without error.\nExpected output file found.' in message:
                factor_dfs.append(df)

        # Combine all successful factor data
        if factor_dfs:
            combined_factors = pd.concat(factor_dfs, axis=1)

            # Remove rows with NaN values
            combined_factors = combined_factors.dropna()
            
            # print(combined_factors)
            return combined_factors
        else:
            print("No valid factor data found to merge.")
            return pd.DataFrame()  # Return an empty DataFrame if no valid data

    def test_docker(self):
        """
        We will mount `env_tpl` into the docker image.
        And run the docker image with `qrun conf.yaml`
        """
        # Store existing directory names in MLRUNS_DIR
        existing_dirs = {d.name for d in MLRUNS_DIR.iterdir() if d.is_dir()}
        print(existing_dirs)

        qtde = QTDockerEnv()
        print("It is running the prepare()")
        qtde.prepare()
        qtde.prepare()  # you can prepare for multiple times. It is expected to handle it correctly
        
        result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="rm -r mlruns", env={"PYTHONPATH": "./"})
        # Run the Qlib backtest
        result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="qrun conf.yaml", env={"PYTHONPATH": "./"})

        # Check for new directories in MLRUNS_DIR
        existing_dirs2 = {d.name for d in MLRUNS_DIR.iterdir() if d.is_dir()}
        print(existing_dirs2)
        new_dirs = {d.name for d in MLRUNS_DIR.iterdir() if d.is_dir()} - existing_dirs
        print(new_dirs)

        if not new_dirs:
            print("No new directories found.")
            return None

        # Access the newly created directory and retrieve port_analysis_1day.pkl
        new_dir = new_dirs.pop()
        pkl_path = MLRUNS_DIR / new_dir / 'artifacts/portfolio_analysis/port_analysis_1day.pkl'
        
        if not pkl_path.exists():
            print(f"File {pkl_path} does not exist.")
            return None

        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)

        return result
