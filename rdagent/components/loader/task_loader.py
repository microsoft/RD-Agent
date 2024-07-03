from rdagent.components.task_implementation.factor_implementation.factor import (
    FactorTask,
)
from rdagent.components.task_implementation.model_implementation.model import ModelTask
from rdagent.core.experiment import Loader


class FactorTaskLoader(Loader[FactorTask]):
    pass


class ModelTaskLoader(Loader[ModelTask]):
    pass
