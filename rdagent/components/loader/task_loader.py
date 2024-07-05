from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.coder.model_coder.model import ModelTask
from rdagent.core.experiment import Loader


class FactorTaskLoader(Loader[FactorTask]):
    pass


class ModelTaskLoader(Loader[ModelTask]):
    pass
