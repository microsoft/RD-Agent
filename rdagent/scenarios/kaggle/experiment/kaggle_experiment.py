from pathlib import Path

from rdagent.components.coder.factor_coder.factor import (
    FactorFBWorkspace,
    FactorTask,
    FeatureExperiment,
)
from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.scenarios.kaggle.experiment.workspace import KGFBWorkspace


class KGModelExperiment(ModelExperiment[ModelTask, KGFBWorkspace, ModelFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = KGFBWorkspace(template_folder_path=Path(__file__).parent / "meta_tpl")


class KGFactorExperiment(FeatureExperiment[FactorTask, KGFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = KGFBWorkspace(template_folder_path=Path(__file__).parent / "meta_tpl")
