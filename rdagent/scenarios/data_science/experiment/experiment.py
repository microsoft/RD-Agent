from copy import deepcopy
from pathlib import Path

from rdagent.core.experiment import Experiment
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.raw_data_loader.exp import (
    DataLoaderFBWorkspace,
    DataLoaderTask,
)
from rdagent.components.coder.factor_coder.factor import (
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.components.coder.model_coder.model import (
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.scenarios.data_science.experiment.workspace import DSFBWorkspace

# KG_MODEL_TYPE_XGBOOST = "XGBoost"
# KG_MODEL_TYPE_RANDOMFOREST = "RandomForest"
# KG_MODEL_TYPE_LIGHTGBM = "LightGBM"
# KG_MODEL_TYPE_NN = "NN"

# KG_MODEL_MAPPING = {
#     KG_MODEL_TYPE_XGBOOST: "model/model_xgboost.py",
#     KG_MODEL_TYPE_RANDOMFOREST: "model/model_randomforest.py",
#     KG_MODEL_TYPE_LIGHTGBM: "model/model_lightgbm.py",
#     KG_MODEL_TYPE_NN: "model/model_nn.py",
# }

# KG_SELECT_MAPPING = {
#     KG_MODEL_TYPE_XGBOOST: "model/select_xgboost.py",
#     KG_MODEL_TYPE_RANDOMFOREST: "model/select_randomforest.py",
#     KG_MODEL_TYPE_LIGHTGBM: "model/select_lightgbm.py",
#     KG_MODEL_TYPE_NN: "model/select_nn.py",
# }



class DataLoaderExperiment(Experiment[DataLoaderTask, DSFBWorkspace, DataLoaderFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = DataLoaderFBWorkspace()


class ModelExperiment(Experiment[ModelTask, DSFBWorkspace, ModelFBWorkspace]):
    def __init__(self, *args, source_feature_size: int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = DSFBWorkspace(
            template_folder_path=Path(__file__).resolve()
            / Path(DS_RD_SETTING.template_path).resolve()
            / DS_RD_SETTING.competition
        )
        if len(self.based_experiments) > 0:
            self.experiment_workspace.inject_code(**self.based_experiments[-1].experiment_workspace.code_dict)
            self.experiment_workspace.data_description = deepcopy(
                self.based_experiments[-1].experiment_workspace.data_description
            )
        else:
            self.experiment_workspace.data_description = [
                (
                    FactorTask(
                        factor_name="Original features",
                        factor_description="The original features",
                        factor_formulation="",
                    ).get_task_information(),
                    source_feature_size,
                )
            ]


class FactorExperiment(Experiment[FactorTask, DSFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, source_feature_size: int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = DSFBWorkspace(
            template_folder_path=Path(__file__).resolve()
            / Path(DS_RD_SETTING.template_path).resolve()
            / DS_RD_SETTING.competition
        )
        if len(self.based_experiments) > 0:
            self.experiment_workspace.inject_code(**self.based_experiments[-1].experiment_workspace.code_dict)
            self.experiment_workspace.data_description = deepcopy(
                self.based_experiments[-1].experiment_workspace.data_description
            )
        else:
            self.experiment_workspace.data_description = [
                (
                    FactorTask(
                        factor_name="Original features",
                        factor_description="The original features",
                        factor_formulation="",
                    ).get_task_information(),
                    source_feature_size,
                )
            ]