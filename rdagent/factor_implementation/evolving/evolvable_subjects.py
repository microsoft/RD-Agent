from __future__ import annotations

import pandas as pd

from core.evolving_framework import EvolvableSubjects
from core.log import FinCoLog
from factor_implementation.share_modules.factor import (
    FactorImplementation,
    FactorImplementationTask,
)


class FactorImplementationList(EvolvableSubjects):
    """
    Factors is a list.
    """

    def __init__(
        self,
        target_factor_tasks: list[FactorImplementationTask],
        corresponding_gt_implementations: list[FactorImplementation] = None,
    ):
        super().__init__()
        self.target_factor_tasks = target_factor_tasks
        self.corresponding_implementations: list[FactorImplementation] = []
        if corresponding_gt_implementations is not None and len(
            corresponding_gt_implementations,
        ) != len(target_factor_tasks):
            self.corresponding_gt_implementations = None
            FinCoLog.warning(
                "The length of corresponding_gt_implementations is not equal to the length of target_factor_tasks, set corresponding_gt_implementations to None",
            )
        else:
            self.corresponding_gt_implementations = corresponding_gt_implementations
