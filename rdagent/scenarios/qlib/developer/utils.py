from typing import List

import pandas as pd

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


def _build_base_feature_workspaces(exp: QlibFactorExperiment) -> list[FactorFBWorkspace]:
    workspaces: list[FactorFBWorkspace] = []
    for file_name, code in exp.base_feature_codes.items():
        workspace = FactorFBWorkspace(
            target_task=FactorTask(
                factor_name=file_name,
                factor_description=f"Base feature from {file_name}",
                factor_formulation="",
            )
        )
        workspace.inject_files(**{"factor.py": code})
        workspaces.append(workspace)
    return workspaces


def _build_execute_calls(exp: QlibFactorExperiment, base_feature_workspaces: list[FactorFBWorkspace]) -> list[tuple]:
    execute_calls = []

    if exp.sub_tasks:
        assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
        execute_calls.extend(
            (implementation.execute, ("All",))
            for implementation, feedback in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
            if implementation and feedback
        )

    execute_calls.extend((workspace.execute, ("All",)) for workspace in base_feature_workspaces)
    return execute_calls


def _resolve_index_level_values(df: pd.DataFrame, level_name: str) -> pd.Index | None:
    matching_levels = [idx for idx, name in enumerate(df.index.names) if name == level_name]
    if not matching_levels:
        return None

    if len(matching_levels) == 1:
        return df.index.get_level_values(matching_levels[0])

    candidate_values = [df.index.get_level_values(idx) for idx in matching_levels]
    first_values = candidate_values[0]
    if all(first_values.equals(values) for values in candidate_values[1:]):
        logger.warning(
            f"Factor dataframe has duplicated '{level_name}' index levels at positions {matching_levels}; "
            "their values are identical, so the first one is used."
        )
        return first_values

    logger.warning(
        f"Skip factor dataframe because index has ambiguous duplicated '{level_name}' levels at positions "
        f"{matching_levels}. index names={list(df.index.names)}"
    )
    return None


def _normalize_factor_index(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalize factor index to a 2-level MultiIndex: (datetime, instrument)."""
    if df is None or df.empty:
        return None

    index_names = list(df.index.names)
    if "datetime" not in index_names:
        return None

    if "instrument" not in index_names:
        logger.warning(f"Skip factor dataframe because index misses 'instrument'. index names={index_names}")
        return None

    datetime_values = _resolve_index_level_values(df, "datetime")
    instrument_values = _resolve_index_level_values(df, "instrument")
    if datetime_values is None or instrument_values is None:
        return None

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


def _process_message_and_df(
    source_name: str,
    message: str,
    df: pd.DataFrame | None,
    factor_dfs: list[pd.DataFrame],
    error_message: str,
) -> str:
    index_info = _format_index_info(df)
    if df is None or "datetime" not in df.index.names:
        logger.warning(f"Factor data from {source_name} has invalid execution output or index: {index_info}")
        logger.warning(f"Factor data from {source_name} is not generated because of {message}")
        return (
            f"{error_message}Factor data from {source_name} is not generated because of {message}. "
            f"index_info={index_info}. "
        )

    normalized_df = _normalize_factor_index(df)
    if normalized_df is None:
        logger.warning(f"Factor data from {source_name} is skipped due to invalid index structure: {index_info}")
        return f"{error_message}Factor data from {source_name} is skipped due to invalid index: {index_info}. "

    time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
    if pd.Timedelta(minutes=1) in time_diff:
        logger.warning(f"Factor data from {source_name} is not generated.")
        return error_message

    factor_dfs.append(normalized_df)
    logger.info(f"Factor data from {source_name} is successfully generated.")
    return error_message


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
        if not isinstance(exp, QlibFactorExperiment):
            continue

        source_name = exp.hypothesis.concise_justification if exp.hypothesis else "BASE factor files"
        base_feature_workspaces = _build_base_feature_workspaces(exp)
        execute_calls = _build_execute_calls(exp, base_feature_workspaces)
        if not execute_calls:
            continue

        message_and_df_list = multiprocessing_wrapper(execute_calls, n=RD_AGENT_SETTINGS.multi_proc_n)
        for message, df in message_and_df_list:
            error_message = _process_message_and_df(source_name, message, df, factor_dfs, error_message)

    # Combine all successful factor data
    if factor_dfs:
        try:
            return pd.concat(factor_dfs, axis=1)
        except Exception as concat_error:
            concat_index_info = " | ".join([f"df#{i}: {_format_index_info(df)}" for i, df in enumerate(factor_dfs)])
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
