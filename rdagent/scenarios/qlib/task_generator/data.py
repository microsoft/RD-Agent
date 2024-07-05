from rdagent.core.task_generator import TaskGenerator
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.utils.env import QTDockerEnv

from pathlib import Path
import pandas as pd
import os
import shutil
import pickle


DIRNAME = Path(__file__).absolute().resolve().parent
RESULT_DIR = Path("/home/finco/quant_result")
MLRUNS_DIR = Path("/home/finco/RDAgent_MS/RD-Agent/rdagent/scenarios/qlib/task_generator/env_factor/mlruns/1")

class QlibFactorRunner(TaskGenerator[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`

    - TODO: implement a qlib handler
    """

    def generate(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        # Process factor data format
        # combined_factors = self.process_factor_data(exp)
        # print("Success in processing factor data.")
        # result = self.test_docker()
        # print(result)
        print(exp)
        # target_dir = self.save_quant_metrics()

        # TODO: Call Docker, pass the combined factors to Docker, and generate backtest results
        pass

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
            
            print(combined_factors)
            return combined_factors
        else:
            print("No valid factor data found to merge.")
            return pd.DataFrame()  # Return an empty DataFrame if no valid data

    # @staticmethod
    # def force_remove_directory(directory_path):
    #     if directory_path.exists():
    #         for root, dirs, files in os.walk(directory_path):
    #             for dir in dirs:
    #                 try:
    #                     os.chmod(os.path.join(root, dir), stat.S_IWUSR | stat.S_IXUSR)
    #                 except PermissionError as e:
    #                     print(f"无法更改目录权限: {os.path.join(root, dir)}: {e}")
    #             for file in files:
    #                 try:
    #                     os.chmod(os.path.join(root, file), stat.S_IWUSR)
    #                 except PermissionError as e:
    #                     print(f"无法更改文件权限: {os.path.join(root, file)}: {e}")
    #         try:
    #             shutil.rmtree(directory_path)
    #             print(f"{directory_path} 已被删除")
    #         except PermissionError as e:
    #             print(f"权限错误：无法删除 {directory_path}: {e}")
    #     else:
    #         print(f"{directory_path} 不存在")


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

    







    # def save_quant_metrics(self):
    #     """
    #     Save the quantitative metrics from the latest result folder.
    #     """
    #     # Ensure the result directory exists
    #     RESULT_DIR.mkdir(parents=True, exist_ok=True)

    #     # Find the latest created directory in the MLRUNS_DIR
    #     latest_dir = max(MLRUNS_DIR.iterdir(), key=os.path.getctime)
    #     artifacts_dir = latest_dir / "artifacts" / "portfolio_analysis"

    #     # Define the target directory to save metrics
    #     target_dir = RESULT_DIR / latest_dir.name
    #     target_dir.mkdir(parents=True, exist_ok=True)

    #     # Copy all files from the artifacts_dir to the target_dir
    #     for item in artifacts_dir.iterdir():
    #         if item.is_file():
    #             shutil.copy(item, target_dir / item.name)
        
    #     print(f"Quantitative metrics have been saved to {target_dir}.")
    #     return target_dir








    # def test_docker(self):
    #     """
    #     We will mount `env_tpl` into the docker image.
    #     And run the docker image with `qrun conf.yaml`
    #     """
    #     qtde = QTDockerEnv()
    #     print("It is running the prepare()")
    #     qtde.prepare()
    #     qtde.prepare()  # you can prepare for multiple times. It is expected to handle it correctly
    #     # the stdout are returned as result
    #     # result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry="qrun conf_mlp.yaml", env={"PYTHONPATH": "/workspace/"})
    #     result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry="qrun conf_mlp.yaml", env={"PYTHONPATH": "./"})
    #     # result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry="ls")

    #     # mlrun_p = DIRNAME / "env_tpl" / "mlruns" 
    #     # self.assertTrue(mlrun_p.exists(), f"Expected output file {mlrun_p} not found")


    # def run_docker(self, factors_file: Path):
    #     """
    #     Run the Docker container to execute the backtest with the combined factors.

    #     Args:
    #         factors_file (Path): The path to the combined factors CSV file.
    #     """
    #     qtde = QTDockerEnv()
    #     qtde.prepare()
    #     qtde.prepare()  # you can prepare for multiple times. It is expected to handle it correctly
        
    #     # Define the Docker command and environment variables
    #     docker_command = [
    #         'docker', 'run', '-v', f'{os.getcwd()}:/workspace', 
    #         'qlib-docker-image', 'python', '/workspace/run_exp.py', str(factors_file)
    #     ]
        
    #     # Run the Docker command
    #     result = subprocess.run(docker_command, capture_output=True, text=True)

    #     if result.returncode != 0:
    #         raise RuntimeError(f'Docker run failed: {result.stderr}')

    #     print(f"Docker run succeeded: {result.stdout}")

    #     # Load the results back into the experiment
    #     result_file = DIRNAME / 'result.json'
    #     if result_file.exists():
    #         with open(result_file, 'r') as f:
    #             result_data = json.load(f)
    #         exp.update_from_dict(result_data)
    #     else:
    #         print("No result file found after Docker run.")

    # def test_docker(self):
    #     """
    #     We will mount `env_tpl` into the docker image.
    #     And run the docker image with `qrun conf.yaml`
    #     """
    #     qtde = QTDockerEnv()
    #     print("It is running the prepare()")
    #     qtde.prepare()
    #     qtde.prepare()  # you can prepare for multiple times. It is expected to handle it correctly
    #     # the stdout are returned as result
    #     result = qtde.run(local_path=str(DIRNAME / "env_tpl"), entry="qrun conf_mlp.yaml", env={"PYTHONPATH": "./"})
    #     print(result)
