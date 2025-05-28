from typing import List

import pandas as pd

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


def process_factor_data(exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
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
        if isinstance(exp, QlibFactorExperiment):
            if len(exp.sub_tasks) > 0:
                # if it has no sub_tasks, the experiment is results from template project.
                # otherwise, it is developed with designed task. So it should have feedback.
                assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
                # Iterate over sub-implementations and execute them to get each factor data
                message_and_df_list = multiprocessing_wrapper(
                    [
                        (implementation.execute, ("All",))
                        for implementation, fb in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
                        if implementation and fb
                    ],  # only execute successfully feedback
                    n=RD_AGENT_SETTINGS.multi_proc_n,
                )
                error_message = ""
                for message, df in message_and_df_list:
                    # Check if factor generation was successful
                    if df is not None and "datetime" in df.index.names:
                        time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                        if pd.Timedelta(minutes=1) not in time_diff:
                            factor_dfs.append(df)
                            logger.info(
                                f"Factor data from {exp.hypothesis.concise_justification} is successfully generated."
                            )
                        else:
                            logger.warning(f"Factor data from {exp.hypothesis.concise_justification} is not generated.")
                    else:
                        error_message += f"Factor data from {exp.hypothesis.concise_justification} is not generated because of {message}"
                        logger.warning(
                            f"Factor data from {exp.hypothesis.concise_justification} is not generated because of {message}"
                        )

    # Combine all successful factor data
    if factor_dfs:
        return pd.concat(factor_dfs, axis=1)
    else:
        raise FactorEmptyError(
            f"No valid factor data found to merge (in process_factor_data) because of {error_message}."
        )
