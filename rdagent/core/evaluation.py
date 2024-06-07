from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
from rdagent.core.task import (
    TaskImplementation,
    FactorImplementTask
)

class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        target_task: FactorImplementTask,
        implementation: TaskImplementation,
        gt_implementation: TaskImplementation,
        **kwargs,
    ):
        raise NotImplementedError
    
class FactorImplementationEvaluator(Evaluator):
    # TODO:
    # I think we should have unified interface for all evaluates, for examples.
    # So we should adjust the interface of other factors
    @abstractmethod
    def evaluate(
        self,
        gt: TaskImplementation,
        gen: TaskImplementation,
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

    def _get_df(self, gt: TaskImplementation, gen: TaskImplementation):
        _, gt_df = gt.execute()
        _, gen_df = gen.execute()
        if isinstance(gen_df, pd.Series):
            gen_df = gen_df.to_frame("source_factor")
        if isinstance(gt_df, pd.Series):
            gt_df = gt_df.to_frame("gt_factor")
        return gt_df, gen_df