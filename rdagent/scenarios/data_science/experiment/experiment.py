from rdagent.core.experiment import Experiment, FBWorkspace

from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.feature_process.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask


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


class DataLoaderExperiment(Experiment[DataLoaderTask, FBWorkspace, FBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = FBWorkspace()

class ModelExperiment(Experiment[ModelTask, FBWorkspace, FBWorkspace]):
    def __init__(self, *args, **kwargs) -> None: # TODO: use previeous step workspace
        super().__init__(*args, **kwargs)
        self.experiment_workspace = FBWorkspace()

class FeatureExperiment(Experiment[FeatureTask, FBWorkspace, FBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = FBWorkspace()

class EnsembleExperiment(Experiment[EnsembleTask, FBWorkspace, FBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = FBWorkspace()

class WorkflowExperiment(Experiment[WorkflowTask, FBWorkspace, FBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = FBWorkspace()

