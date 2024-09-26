from copy import deepcopy
from pathlib import Path

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
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
        self.experiment_workspace = KGFBWorkspace(
            template_folder_path=Path(__file__).parent / f"{KAGGLE_IMPLEMENT_SETTING.competition}_template"
        )
        if len(self.based_experiments) > 0:
            self.experiment_workspace.inject_code(**self.based_experiments[-1].experiment_workspace.code_dict)
            self.experiment_workspace.data_description = deepcopy(
                self.based_experiments[-1].experiment_workspace.data_description
            )
            self.experiment_workspace.model_description = deepcopy(
                self.based_experiments[-1].experiment_workspace.model_description
            )


class KGFactorExperiment(FeatureExperiment[FactorTask, KGFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = KGFBWorkspace(
            template_folder_path=Path(__file__).parent / f"{KAGGLE_IMPLEMENT_SETTING.competition}_template"
        )
        if len(self.based_experiments) > 0:
            self.experiment_workspace.inject_code(**self.based_experiments[-1].experiment_workspace.code_dict)
            self.experiment_workspace.data_description = deepcopy(
                self.based_experiments[-1].experiment_workspace.data_description
            )
            self.experiment_workspace.model_description = deepcopy(
                self.based_experiments[-1].experiment_workspace.model_description
            )
