from typing import List
import pandas as pd

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelFBWorkspace
from rdagent.components.runner import CachedRunner
from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.feature_engineering.experiment.feature_experiment import FEFeatureExperiment


class FEFeatureRunner(CachedRunner[FEFeatureExperiment]):
    def develop(self, exp: FEFeatureExperiment) -> FEFeatureExperiment:
        import pickle
        with open('/home/v-yuanteli/RD-Agent/git_ignore_folder/test_featexp_data.pkl', 'wb') as f:
            pickle.dump(exp, f)
        print("Feature Experiment object saved to test_featexp_data.pkl")


        # 这里考虑把每次实验都加一次原始数据
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        if RUNNER_SETTINGS.cache_result:
            cache_hit, result = self.get_cache_result(exp)
            if cache_hit:
                exp.result = result
                return exp
        
        #TODO 这里对应于SOTA因子库的概念
        if exp.based_experiments:
            SOTA_factor = None
            if len(exp.based_experiments) > 1:
                SOTA_factor = self.process_factor_data(exp.based_experiments)

            # Process the new factors data
            new_factors = self.process_factor_data(exp)

            if new_factors.empty:
                raise FactorEmptyError("No valid factor data found to merge.")

            # Combine the SOTA factor and new factors if SOTA factor exists
            if SOTA_factor is not None and not SOTA_factor.empty:
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                if new_factors.empty:
                    raise FactorEmptyError("No valid factor data found to merge.")
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors

            # Sort and nest the combined factors under 'feature'
            # TODO 这里是去重吧，针对feature的格式处理 kaggle应该不需要
            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns

            # Save the combined factors to the workspace
            with open(exp.experiment_workspace.workspace_path / "combined_factors_df.pkl", "wb") as f:
                pickle.dump(combined_factors, f)

        # TODO 这里还是execute，应该是连kaggle的dockers
        result = exp.experiment_workspace.execute(
            qlib_config_name=f"conf.yaml" if len(exp.based_experiments) == 0 else "conf_combined.yaml"
        )

        exp.result = result
        if RUNNER_SETTINGS.cache_result:
            self.dump_cache_result(exp, result)

        return exp
        

        return exp

    def process_factor_data(self, exp_or_list: List[FEFeatureExperiment] | FEFeatureExperiment) -> pd.DataFrame:
        """
        Process and combine factor data from experiment implementations.

        Args:
            exp (ASpecificExp): The experiment containing factor data.

        Returns:
            pd.DataFrame: Combined factor data without NaN values.
        """
        #TODO 这里需要把task的代码执行一遍，得到一个dataframe
        if isinstance(exp_or_list, FEFeatureExperiment):
            exp_or_list = [exp_or_list]
        factor_dfs = []

        # Collect all exp's dataframes
        for exp in exp_or_list:
            # Iterate over sub-implementations and execute them to get each factor data
            #TODO 这里应当使用feature_execute函数实现
            message_and_df_list = multiprocessing_wrapper(
                [(implementation.feature_execute) for implementation in exp.sub_workspace_list],
                n=RD_AGENT_SETTINGS.multi_proc_n,
            )
            #TODO datatime这些 这里应该不需要了
            for message, df in message_and_df_list:
                # Check if factor generation was successful
                if df is not None and "datetime" in df.index.names:
                    time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                    if pd.Timedelta(minutes=1) not in time_diff:
                        factor_dfs.append(df)

        # Combine all successful factor data
        if factor_dfs:
            return pd.concat(factor_dfs, axis=1)
        else:
            raise FactorEmptyError("No valid factor data found to merge.")