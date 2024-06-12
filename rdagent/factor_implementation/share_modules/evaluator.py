import json
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from finco.log import FinCoLog
from jinja2 import Template
from rdagent.factor_implementation.share_modules.factor_implementation_config import (
    FactorImplementSettings,
)
from rdagent.oai.llm_utils import APIBackend

from factor_implementation.share_modules.factor import (
    FactorImplementation,
    FactorImplementationTask,
)
from factor_implementation.share_modules.prompt import FactorImplementationPrompts


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        target_task: FactorImplementationTask,
        implementation: FactorImplementation,
        gt_implementation: FactorImplementation,
        **kwargs,
    ):
        raise NotImplementedError


class FactorImplementationCodeEvaluator(Evaluator):
    def evaluate(
        self,
        target_task: FactorImplementationTask,
        implementation: FactorImplementation,
        execution_feedback: str,
        factor_value_feedback: str = "",
        gt_implementation: FactorImplementation = None,
        **kwargs,
    ):
        factor_information = target_task.get_factor_information()
        code = implementation.code

        system_prompt = FactorImplementationPrompts()["evaluator_code_feedback_v1_system"]

        execution_feedback_to_render = execution_feedback
        user_prompt = Template(
            FactorImplementationPrompts()["evaluator_code_feedback_v1_user"],
        ).render(
            factor_information=factor_information,
            code=code,
            execution_feedback=execution_feedback_to_render,
            factor_value_feedback=factor_value_feedback,
            gt_code=gt_implementation.code if gt_implementation else None,
        )
        while (
            APIBackend().build_messages_and_calculate_token(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                former_messages=[],
            )
            > FactorImplementSettings().chat_token_limit
        ):
            execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            user_prompt = Template(
                FactorImplementationPrompts()["evaluator_code_feedback_v1_user"],
            ).render(
                factor_information=factor_information,
                code=code,
                execution_feedback=execution_feedback_to_render,
                factor_value_feedback=factor_value_feedback,
                gt_code=gt_implementation.code if gt_implementation else None,
            )
        critic_response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )

        # critic_response = json.loads(critic_response)
        return critic_response


class FactorImplementationEvaluator(Evaluator):
    # TODO:
    # I think we should have unified interface for all evaluates, for examples.
    # So we should adjust the interface of other factors
    @abstractmethod
    def evaluate(
        self,
        gt: FactorImplementation,
        gen: FactorImplementation,
    ) -> Tuple[str, object]:
        """You can get the dataframe by

        .. code-block:: python

            _, gt_df = gt.execute()
            _, gen_df = gen.execute()

        Returns
        -------
        Tuple[str, object]
            - str: the text-based description of the evaluation result
            - object: a comparable metric (bool, integer, float ...)

        """
        raise NotImplementedError("Please implement the `evaluator` method")

    def _get_df(self, gt: FactorImplementation, gen: FactorImplementation):
        _, gt_df = gt.execute()
        _, gen_df = gen.execute()
        if isinstance(gen_df, pd.Series):
            gen_df = gen_df.to_frame("source_factor")
        if isinstance(gt_df, pd.Series):
            gt_df = gt_df.to_frame("gt_factor")
        return gt_df, gen_df


# NOTE: the following evaluators are splited from FactorImplementationValueEvaluator


class FactorImplementationSingleColumnEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: FactorImplementation,
        gen: FactorImplementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        if len(gen_df.columns) == 1 and len(gt_df.columns) == 1:
            return "Both dataframes have only one column.", True
        elif len(gen_df.columns) != 1:
            gen_df = gen_df.iloc(axis=1)[
                [
                    0,
                ]
            ]
            return (
                "The source dataframe has more than one column. Please check the implementation. We only evaluate the first column.",
                False,
            )
        return "", False

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationIndexFormatEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: FactorImplementation,
        gen: FactorImplementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)
        idx_name_right = gen_df.index.names == ("datetime", "instrument")
        if idx_name_right:
            return (
                'The index of the dataframe is ("datetime", "instrument") and align with the predefined format.',
                True,
            )
        else:
            return (
                'The index of the dataframe is not ("datetime", "instrument"). Please check the implementation.',
                False,
            )

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationRowCountEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: FactorImplementation,
        gen: FactorImplementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        if gen_df.shape[0] == gt_df.shape[0]:
            return "Both dataframes have the same rows count.", True
        else:
            return (
                f"The source dataframe and the ground truth dataframe have different rows count. The source dataframe has {gen_df.shape[0]} rows, while the ground truth dataframe has {gt_df.shape[0]} rows. Please check the implementation.",
                False,
            )

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationIndexEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: FactorImplementation,
        gen: FactorImplementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        if gen_df.index.equals(gt_df.index):
            return "Both dataframes have the same index.", True
        else:
            return (
                "The source dataframe and the ground truth dataframe have different index. Please check the implementation.",
                False,
            )

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationMissingValuesEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: FactorImplementation,
        gen: FactorImplementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        if gen_df.isna().sum().sum() == gt_df.isna().sum().sum():
            return "Both dataframes have the same missing values.", True
        else:
            return (
                f"The dataframes do not have the same missing values. The source dataframe has {gen_df.isna().sum().sum()} missing values, while the ground truth dataframe has {gt_df.isna().sum().sum()} missing values. Please check the implementation.",
                False,
            )

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationValuesEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: FactorImplementation,
        gen: FactorImplementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        try:
            close_values = gen_df.sub(gt_df).abs().lt(1e-6)
            result_int = close_values.astype(int)
            pos_num = result_int.sum().sum()
            acc_rate = pos_num / close_values.size
        except:
            close_values = gen_df
        if close_values.all().iloc[0]:
            return (
                "All values in the dataframes are equal within the tolerance of 1e-6.",
                acc_rate,
            )
        else:
            return (
                "Some values differ by more than the tolerance of 1e-6. Check for rounding errors or differences in the calculation methods.",
                acc_rate,
            )

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationCorrelationEvaluator(FactorImplementationEvaluator):
    def __init__(self, hard_check: bool) -> None:
        self.hard_check = hard_check

    def evaluate(
        self,
        gt: FactorImplementation,
        gen: FactorImplementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        concat_df = pd.concat([gen_df, gt_df], axis=1)
        concat_df.columns = ["source", "gt"]
        ic = concat_df.groupby("datetime").apply(lambda df: df["source"].corr(df["gt"])).dropna().mean()
        ric = (
            concat_df.groupby("datetime")
            .apply(lambda df: df["source"].corr(df["gt"], method="spearman"))
            .dropna()
            .mean()
        )

        if self.hard_check:
            if ic > 0.99 and ric > 0.99:
                return (
                    f"The dataframes are highly correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}.",
                    True,
                )
            else:
                return (
                    f"The dataframes are not sufficiently high correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}. Investigate the factors that might be causing the discrepancies and ensure that the logic of the factor calculation is consistent.",
                    False,
                )
        else:
            return f"The ic is ({ic:.6f}) and the rankic is ({ric:.6f}).", ic

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationValEvaluator(FactorImplementationEvaluator):
    def evaluate(self, gt: FactorImplementation, gen: FactorImplementation):
        _, gt_df = gt.execute()
        _, gen_df = gen.execute()
        # FIXME: refactor the two classes
        fiv = FactorImplementationValueEvaluator()
        return fiv.evaluate(source_df=gen_df, gt_df=gt_df)

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationValueEvaluator(Evaluator):
    # TODO: let's discuss the about the interface of the evaluator
    def evaluate(
        self,
        source_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        **kwargs,
    ) -> Tuple:
        conclusions = []

        if isinstance(source_df, pd.Series):
            source_df = source_df.to_frame("source_factor")
            conclusions.append(
                "The source dataframe is a series, better convert it to a dataframe.",
            )
        if gt_df is not None and isinstance(gt_df, pd.Series):
            gt_df = gt_df.to_frame("gt_factor")
            conclusions.append(
                "The ground truth dataframe is a series, convert it to a dataframe.",
            )

        # Check if both dataframe has only one columns
        if len(source_df.columns) == 1:
            conclusions.append("The source dataframe has only one column which is correct.")
        else:
            conclusions.append(
                "The source dataframe has more than one column. Please check the implementation. We only evaluate the first column.",
            )
            source_df = source_df.iloc(axis=1)[
                [
                    0,
                ]
            ]

        if list(source_df.index.names) != ["datetime", "instrument"]:
            conclusions.append(
                rf"The index of the dataframe is not (\"datetime\", \"instrument\"), instead is {source_df.index.names}. Please check the implementation.",
            )
        else:
            conclusions.append(
                'The index of the dataframe is ("datetime", "instrument") and align with the predefined format.',
            )

        # Check if both dataframe have the same rows count
        if gt_df is not None:
            if source_df.shape[0] == gt_df.shape[0]:
                conclusions.append("Both dataframes have the same rows count.")
                same_row_count_result = True
            else:
                conclusions.append(
                    f"The source dataframe and the ground truth dataframe have different rows count. The source dataframe has {source_df.shape[0]} rows, while the ground truth dataframe has {gt_df.shape[0]} rows. Please check the implementation.",
                )
                same_row_count_result = False

            # Check whether both dataframe has the same index
            if source_df.index.equals(gt_df.index):
                conclusions.append("Both dataframes have the same index.")
                same_index_result = True
            else:
                conclusions.append(
                    "The source dataframe and the ground truth dataframe have different index. Please check the implementation.",
                )
                same_index_result = False

            # Check for the same missing values (NaN)
            if source_df.isna().sum().sum() == gt_df.isna().sum().sum():
                conclusions.append("Both dataframes have the same missing values.")
                same_missing_values_result = True
            else:
                conclusions.append(
                    f"The dataframes do not have the same missing values. The source dataframe has {source_df.isna().sum().sum()} missing values, while the ground truth dataframe has {gt_df.isna().sum().sum()} missing values. Please check the implementation.",
                )
                same_missing_values_result = False

            # Check if the values are the same within a small tolerance
            if not same_index_result:
                conclusions.append(
                    "The source dataframe and the ground truth dataframe have different index. Give up comparing the values and correlation because it's useless",
                )
                same_values_result = False
                high_correlation_result = False
            else:
                close_values = source_df.sub(gt_df).abs().lt(1e-6)
                if close_values.all().iloc[0]:
                    conclusions.append(
                        "All values in the dataframes are equal within the tolerance of 1e-6.",
                    )
                    same_values_result = True
                else:
                    conclusions.append(
                        "Some values differ by more than the tolerance of 1e-6. Check for rounding errors or differences in the calculation methods.",
                    )
                    same_values_result = False

                # Check the ic and rankic between the two dataframes
                concat_df = pd.concat([source_df, gt_df], axis=1)
                concat_df.columns = ["source", "gt"]
                try:
                    ic = concat_df.groupby("datetime").apply(lambda df: df["source"].corr(df["gt"])).dropna().mean()
                    ric = (
                        concat_df.groupby("datetime")
                        .apply(lambda df: df["source"].corr(df["gt"], method="spearman"))
                        .dropna()
                        .mean()
                    )

                    if ic > 0.99 and ric > 0.99:
                        conclusions.append(
                            f"The dataframes are highly correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}.",
                        )
                        high_correlation_result = True
                    else:
                        conclusions.append(
                            f"The dataframes are not sufficiently high correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}. Investigate the factors that might be causing the discrepancies and ensure that the logic of the factor calculation is consistent.",
                        )
                        high_correlation_result = False

                    # Check for shifted alignments only in the "datetime" index
                    max_shift_days = 2
                    for shift in range(-max_shift_days, max_shift_days + 1):
                        if shift == 0:
                            continue  # Skip the case where there is no shift

                        shifted_source_df = source_df.groupby(level="instrument").shift(shift)
                        concat_df = pd.concat([shifted_source_df, gt_df], axis=1)
                        concat_df.columns = ["source", "gt"]
                        shifted_ric = (
                            concat_df.groupby("datetime")
                            .apply(lambda df: df["source"].corr(df["gt"], method="spearman"))
                            .dropna()
                            .mean()
                        )
                        if shifted_ric > 0.99:
                            conclusions.append(
                                f"The dataframes are highly correlated with a shift of {max_shift_days} days in the 'date' index. Shifted rankic: {shifted_ric:.6f}.",
                            )
                            break
                    else:
                        conclusions.append(
                            f"No sufficient correlation found when shifting up to {max_shift_days} days in the 'date' index. Investigate the factors that might be causing discrepancies.",
                        )

                except Exception as e:
                    FinCoLog().warning(f"Error occurred when calculating the correlation: {e!s}")
                    conclusions.append(
                        f"Some error occurred when calculating the correlation. Investigate the factors that might be causing the discrepancies and ensure that the logic of the factor calculation is consistent. Error: {e}",
                    )
                    high_correlation_result = False

        # Combine all conclusions into a single string
        conclusion_str = "\n".join(conclusions)

        final_result = (same_values_result or high_correlation_result) if gt_df is not None else False
        return conclusion_str, final_result


# TODO:
def shorten_prompt(tpl: str, render_kwargs: dict, shorten_key: str, max_trail: int = 10) -> str:
    """When the prompt is too long. We have to shorten it.
    But we should not truncate the prompt directly, so we should find the key we want to shorten and then shorten it.
    """
    # TODO: this should replace most of code in
    # - FactorImplementationFinalDecisionEvaluator.evaluate
    # - FactorImplementationCodeEvaluator.evaluate


class FactorImplementationFinalDecisionEvaluator(Evaluator):
    def evaluate(
        self,
        target_task: FactorImplementationTask,
        execution_feedback: str,
        value_feedback: str,
        code_feedback: str,
        **kwargs,
    ) -> Tuple:
        system_prompt = FactorImplementationPrompts()["evaluator_final_decision_v1_system"]
        execution_feedback_to_render = execution_feedback
        user_prompt = Template(
            FactorImplementationPrompts()["evaluator_final_decision_v1_user"],
        ).render(
            factor_information=target_task.get_factor_information(),
            execution_feedback=execution_feedback_to_render,
            code_feedback=code_feedback,
            factor_value_feedback=(
                value_feedback
                if value_feedback is not None
                else "No Ground Truth Value provided, so no evaluation on value is performed."
            ),
        )
        while (
            APIBackend().build_messages_and_calculate_token(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                former_messages=[],
            )
            > FactorImplementSettings().chat_token_limit
        ):
            execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            user_prompt = Template(
                FactorImplementationPrompts()["evaluator_final_decision_v1_user"],
            ).render(
                factor_information=target_task.get_factor_information(),
                execution_feedback=execution_feedback_to_render,
                code_feedback=code_feedback,
                factor_value_feedback=(
                    value_feedback
                    if value_feedback is not None
                    else "No Ground Truth Value provided, so no evaluation on value is performed."
                ),
            )

        final_evaluation_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            ),
        )
        return (
            final_evaluation_dict["final_decision"],
            final_evaluation_dict["final_feedback"],
        )
