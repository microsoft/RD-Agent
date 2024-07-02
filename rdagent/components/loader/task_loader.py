from rdagent.components.task_implementation.factor_implementation.factor import (
    FactorTask,
)
from rdagent.core.experiment import Loader


class FactorTaskLoader(Loader[FactorTask]):
    pass


class ModelTaskLoader(Loader[FactorTask]):
    pass
