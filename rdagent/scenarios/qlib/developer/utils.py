from typing import List

import pandas as pd

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

def _normalize_factor_index(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalize factor index to a 2-level MultiIndex: (datetime, instrument)."""
    if df is None or df.empty:
        return None

    index_names = list(df.index.names)
    if "datetime" not in index_names:
        return None

    if "instrument" not in index_names:
        logger.warning(
            f"Skip factor dataframe because index misses 'instrument'. index names={index_names}"
        )
        return None

    datetime_values = df.index.get_level_values("datetime")
    instrument_values = df.index.get_level_values("instrument")
    normalized = df.copy()
    normalized.index = pd.MultiIndex.from_arrays(
        [datetime_values, instrument_values],
        names=["datetime", "instrument"],
    )
    return normalized


def _format_index_info(df: pd.DataFrame | None) -> str:
    if df is None:
        return "df is None"
    return f"index_type={type(df.index).__name__}, nlevels={df.index.nlevels}, names={list(df.index.names)}"


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
    error_message = ""

    # Collect all exp's dataframes
    for exp in exp_or_list:
        if isinstance(exp, QlibFactorExperiment):
            base_feature_workspaces = []
            source_name = exp.hypothesis.concise_justification if exp.hypothesis else "BASE factor files"
            # Build runnable workspaces from externally provided base feature code files.
            for file_name, code in exp.base_feature_codes.items():
                ws = FactorFBWorkspace(
                    target_task=FactorTask(
                        factor_name=file_name,
                        factor_description=f"Base feature from {file_name}",
                        factor_formulation="",
                    )
                )
                ws.inject_files(**{"factor.py": code})
                base_feature_workspaces.append(ws)

            execute_calls = []
            if len(exp.sub_tasks) > 0:
                # if it has no sub_tasks, the experiment is results from template project.
                # otherwise, it is developed with designed task. So it should have feedback.
                assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
                # Iterate over sub-implementations and execute them to get each factor data
                execute_calls.extend(
                    [
                        (implementation.execute, ("All",))
                        for implementation, fb in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
                        if implementation and fb
                    ]
                )  # only execute successfully feedback
            execute_calls.extend((workspace.execute, ("All",)) for workspace in base_feature_workspaces)

            if execute_calls:
                message_and_df_list = multiprocessing_wrapper(execute_calls, n=RD_AGENT_SETTINGS.multi_proc_n)
                for message, df in message_and_df_list:
                    # Check if factor generation was successful
                    if df is not None and "datetime" in df.index.names:
                        normalized_df = _normalize_factor_index(df)
                        if normalized_df is None:
                            logger.warning(
                                f"Factor data from {source_name} is skipped due to invalid index structure: {_format_index_info(df)}"
                            )
                            error_message += (
                                "Factor data from "
                                f"{source_name} is skipped due to invalid index: "
                                f"{_format_index_info(df)}. "
                            )
                            continue
                        time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                        if pd.Timedelta(minutes=1) not in time_diff:
                            factor_dfs.append(normalized_df)
                            logger.info(
                                f"Factor data from {source_name} is successfully generated."
                            )
                        else:
                            logger.warning(f"Factor data from {source_name} is not generated.")
                    else:
                        logger.warning(
                            f"Factor data from {source_name} has invalid execution output or index: {_format_index_info(df)}"
                        )
                        error_message += (
                            f"Factor data from {source_name} is not generated because of "
                            f"{message}. index_info={_format_index_info(df)}. "
                        )
                        logger.warning(
                            f"Factor data from {source_name} is not generated because of {message}"
                        )

    # Combine all successful factor data
    if factor_dfs:
        try:
            return pd.concat(factor_dfs, axis=1)
        except Exception as concat_error:
            concat_index_info = " | ".join(
                [f"df#{i}: {_format_index_info(df)}" for i, df in enumerate(factor_dfs)]
            )
            logger.warning(
                f"Failed to concat factor data due to index misalignment. concat_error={concat_error}; collected_index_info={concat_index_info}"
            )
            raise FactorEmptyError(
                "Failed to concat factor data due to index misalignment or incompatible index structure. "
                f"concat_error={concat_error}; collected_index_info={concat_index_info}; details={error_message}"
            ) from concat_error
    else:
        raise FactorEmptyError(
            f"No valid factor data found to merge (in process_factor_data) because of {error_message}."
        )
