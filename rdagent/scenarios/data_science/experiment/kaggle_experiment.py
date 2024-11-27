from copy import deepcopy
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
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
from rdagent.components.coder.data_science.raw_data_loader.raw_data_loader import (
    DataLoaderExperiment,
    DataLoaderFBWorkspace,
    DataLoaderTask,
)

from rdagent.scenarios.data_science.experiment.workspace import KGFBWorkspace

KG_MODEL_TYPE_XGBOOST = "XGBoost"
KG_MODEL_TYPE_RANDOMFOREST = "RandomForest"
KG_MODEL_TYPE_LIGHTGBM = "LightGBM"
KG_MODEL_TYPE_NN = "NN"

KG_MODEL_MAPPING = {
    KG_MODEL_TYPE_XGBOOST: "model/model_xgboost.py",
    KG_MODEL_TYPE_RANDOMFOREST: "model/model_randomforest.py",
    KG_MODEL_TYPE_LIGHTGBM: "model/model_lightgbm.py",
    KG_MODEL_TYPE_NN: "model/model_nn.py",
}

KG_SELECT_MAPPING = {
    KG_MODEL_TYPE_XGBOOST: "model/select_xgboost.py",
    KG_MODEL_TYPE_RANDOMFOREST: "model/select_randomforest.py",
    KG_MODEL_TYPE_LIGHTGBM: "model/select_lightgbm.py",
    KG_MODEL_TYPE_NN: "model/select_nn.py",
}

class KGDataLoaderExperiment(DataLoaderExperiment[ModelTask, KGFBWorkspace, ModelFBWorkspace]):
    # TODO: complete the implementation
    def __init__(self, *args, source_feature_size: int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: It seems there are some problems as the folder has not been created.
        # self.experiment_workspace = KGFBWorkspace(
        #     template_folder_path=Path(__file__).resolve()
        #     / Path(DS_RD_SETTING.template_path).resolve()
        #     / DS_RD_SETTING.competition
        # )


class KGModelExperiment(ModelExperiment[ModelTask, KGFBWorkspace, ModelFBWorkspace]):
    def __init__(self, *args, source_feature_size: int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = KGFBWorkspace(
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


class KGFactorExperiment(FeatureExperiment[FactorTask, KGFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, source_feature_size: int = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = KGFBWorkspace(
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
