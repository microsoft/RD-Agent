from rdagent.components.task_implementation.factor_implementation.factor import (
    FactorExperiment,
)
from rdagent.core.experiment import Loader


class FactorExperimentLoader(Loader[FactorExperiment]):
    pass


class ModelExperimentLoader(Loader[FactorExperiment]):
    pass
