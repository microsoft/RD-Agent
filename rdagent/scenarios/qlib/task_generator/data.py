from pathlib import Path
import shutil
from typing import List
import pandas as pd
import pickle
from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.core.task_generator import TaskGenerator
from rdagent.utils.env import QTDockerEnv, LocalConf, LocalEnv
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.core.log import RDAgentLog

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()
logger = RDAgentLog()

# class QlibFactorExpWorkspace:
           
#     def prepare():
#         # create a folder;
#         # copy template
#         # place data inside the folder `combined_factors`
#         #
#     def execute():
#         de = DockerEnv()
#         de.run(local_path=self.ws_path, entry="qrun conf.yaml")

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
    
    def FetchAlpha158ResultFromDocker(self):
        """
         Run Docker to get alpha158 result.

        This method prepares the Qlib Docker environment, executes the necessary commands to 
        run the backtest, and fetches the results stored in a pickle file.

        Returns:
            Any: The alpha158 result. If successful, returns a pandas DataFrame. Otherwise, returns None.
        """
        # Initialize and prepare the Qlib Docker environment
        qtde = QTDockerEnv()
        qtde.prepare()
        
        # Clean up any previous run artifacts by deleting the mlruns directory
        result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="rm -r mlruns", env={"PYTHONPATH": "./"})
        
        # Run the Qlib backtest using the configuration file conf.yaml
        result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="qrun conf.yaml", env={"PYTHONPATH": "./"})
        
        # Execute a Python script to extract the experiment results
        result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="python read_exp_res.py")

        pkl_path = DIRNAME / 'env_factor/qlib_res.pkl'

        if not pkl_path.exists():
            logger.error(f"File {pkl_path} does not exist.")
            return None

        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)

        # Check if the loaded result is a pandas DataFrame and not empty
        if isinstance(result, pd.DataFrame):
            if not result.empty:
                logger.info("Successfully retrieved alpha158 result.")
                return result
            else:
                logger.error("Result DataFrame is empty.")
                return None
        else:
            logger.error("Data format error.")
            return None


    def generate(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        """
        Generate the experiment by processing and combining factor data,
        then passing the combined data to Docker for backtest results.
        """

        SOTA_factor = self.process_factor_data(exp.based_experiments)
        
        if exp.based_experiments[-1].result is None:
            exp.based_experiments[-1].result = self.FetchAlpha158ResultFromDocker()
        
        # Process the new factors data
        new_factors = self.process_factor_data(exp)
        
        # Combine the SOTA factor and new factors if SOTA factor exists
        if SOTA_factor is not None:
            combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
        else:
            combined_factors = new_factors
        
        # Sort and nest the combined factors under 'feature'
        combined_factors = combined_factors.sort_index()
        new_columns = pd.MultiIndex.from_product([['feature'], combined_factors.columns])
        combined_factors.columns = new_columns

        # Save the combined factors to a pickle file
        combined_factors_path = DIRNAME / 'env_factor/combined_factors_df.pkl'
        with open(combined_factors_path, 'wb') as f:
            pickle.dump(combined_factors, f)

        """ Docker run
        # Call Docker, pass the combined factors to Docker, and generate backtest results
        qtde = QTDockerEnv()
        qtde.prepare()
        
        # Run the Docker command
        result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="rm -r mlruns", env={"PYTHONPATH": "./"})
        # Run the Qlib backtest
        result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="qrun conf_combined.yaml", env={"PYTHONPATH": "./"})

        result = qtde.run(local_path=str(DIRNAME / "env_factor"), entry="python read_exp_res.py")

        pkl_path = DIRNAME / 'env_factor/qlib_res.pkl'

        if not pkl_path.exists():
            logger.error(f"File {pkl_path} does not exist.")
            return None

        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
         """
        
        # Local run
        # Clean up any previous run artifacts by deleting the mlruns directory
        mlruns_path = DIRNAME_local / 'mlruns' / '1'
        if mlruns_path.exists() and mlruns_path.is_dir():
            shutil.rmtree(mlruns_path)

        # Prepare local Qlib environment
        local_conf = LocalConf(
            py_bin=PROP_SETTING.py_bin,
            default_entry="qrun conf_combined.yaml",
        )
        qle = LocalEnv(conf=local_conf)
        qle.prepare()
        conf_path = str(DIRNAME / "env_factor" / "conf_combined.yaml") 
        qle.run(entry="qrun " + conf_path)

        # Verify if the new folder is created
        mlrun_p = DIRNAME_local / 'mlruns' / '1' 
        assert mlrun_p.exists(), f"Expected output file {mlrun_p} not found"

        # Locate the newly generated folder in mlruns/1/
        new_folders = [folder for folder in mlrun_p.iterdir() if folder.is_dir()]
        if not new_folders:
            raise FileNotFoundError("No new folders found in 'mlruns/1/'.")

        new_folder = new_folders[0]  # Assuming there's only one new folder
        pickle_file = new_folder / 'artifacts' / 'portfolio_analysis' / 'port_analysis_1day.pkl'
        assert pickle_file.exists(), f"Expected pickle file {pickle_file} not found"

        with open(pickle_file, 'rb') as f:
            result = pickle.load(f)
        
        exp.result = result
        
        # Check if the result is valid and is a DataFrame
        if isinstance(result, pd.DataFrame):
            if not result.empty:
                logger.info("Successfully retrieved experiment result.")
                return exp
            else:
                logger.error("Result DataFrame is empty.")
                return None
        else:
            logger.error("Data format error.")
            return None

    def process_factor_data(self, exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
        """
        Process and combine factor data from experiment implementations.

        Args:
            exp (ASpecificExp): The experiment containing factor data.

        Returns:
            pd.DataFrame: Combined factor data without NaN values.
        """
        if isinstance(exp_or_list, QlibFactorExperiment):
            exp_or_list = [exp_or_list]
        factor_dfs = []

        # Collect all exp's dataframes
        for exp in exp_or_list:
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
            logger.error("No valid factor data found to merge.")
            return pd.DataFrame()  # Return an empty DataFrame if no valid data
