import json
import pickle
import site
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from rdagent.components.coder.model_coder.conf import MODEL_IMPL_SETTINGS
from rdagent.core.exception import CodeFormatError
from rdagent.core.experiment import Experiment, FBWorkspace, Task
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils import get_module_by_module_path


class ModelTask(Task):
    def __init__(
        self, name: str, description: str, formulation: str, architecture: str, variables: Dict[str, str], model_type: Optional[str] = None
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.formulation: str = formulation
        self.architecture: str = architecture
        self.variables: str = variables
        self.model_type: str = model_type  # Tabular for tabular model, TimesSeries for time series model, Graph for graph model 

    def get_task_information(self):
        return f"""name: {self.name}
description: {self.description}
formulation: {self.formulation}
architecture: {self.architecture}
variables: {self.variables}
model_type: {self.model_type}
"""

    @staticmethod
    def from_dict(dict):
        return ModelTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


class ModelFBWorkspace(FBWorkspace):
    """
    It is a Pytorch model implementation task;
    All the things are placed in a folder.

    Folder
    - data source and documents prepared by `prepare`
        - Please note that new data may be passed in dynamically in `execute`
    - code (file `model.py` ) injected by `inject_code`
        - the `model.py` that contains a variable named `model_cls` which indicates the implemented model structure
            - `model_cls` is a instance of `torch.nn.Module`;


    We'll import the model in the implementation in file `model.py` after setting the cwd into the directory
    - from model import model_cls
    - initialize the model by initializing it `model_cls(input_dim=INPUT_DIM)`
    - And then verify the model.

    """

    def execute(
        self,
        batch_size: int = 8,
        num_features: int = 10,
        num_timesteps: int = 4,
        num_nodes: int = 50,
        num_edges: int = 100,
        input_value: float = 1.0,
        param_init_value: float = 1.0,
    ):
        super().execute()
        try:
            if MODEL_IMPL_SETTINGS.enable_execution_cache:
                # NOTE: cache the result for the same code
                target_file_name = md5_hash(
                    f"{batch_size}_{num_features}_{num_timesteps}_{input_value}_{param_init_value}_{self.code_dict['model.py']}"
                )
                cache_file_path = Path(MODEL_IMPL_SETTINGS.cache_location) / f"{target_file_name}.pkl"
                Path(MODEL_IMPL_SETTINGS.cache_location).mkdir(exist_ok=True, parents=True)
                if cache_file_path.exists():
                    return pickle.load(open(cache_file_path, "rb"))
            mod = get_module_by_module_path(str(self.workspace_path / "model.py"))
            model_cls = mod.model_cls

            if self.target_task.model_type == "Tabular":
                input_shape = (batch_size, num_features)
                m = model_cls(num_features=input_shape[1])
                data = torch.full(input_shape, input_value)
            elif self.target_task.model_type == "TimeSeries":
                input_shape = (batch_size, num_features, num_timesteps)
                m = model_cls(num_features=input_shape[1], num_timesteps=input_shape[2])
                data = torch.full(input_shape, input_value)
            elif self.target_task.model_type == "Graph":
                node_feature = torch.randn(batch_size, num_nodes, num_features)
                edge_index = torch.randint(0, num_nodes, (2, num_edges))
                m = model_cls(num_nodes=num_nodes, num_features=num_features)
                data = (node_feature, edge_index)
            else:
                raise ValueError(f"Unsupported model type: {self.target_task.model_type}")

            # Initialize all parameters of `m` to `param_init_value`
            for _, param in m.named_parameters():
                param.data.fill_(param_init_value)
            
            # Execute the model
            if self.target_task.model_type == "Graph":
                out = m(*data)
            else:
                out = m(data)
            
            execution_model_output = out.cpu().detach()
            execution_feedback_str = f"Execution successful, output tensor shape: {execution_model_output.shape}"
            
            if MODEL_IMPL_SETTINGS.enable_execution_cache:
                pickle.dump((execution_feedback_str, execution_model_output), open(cache_file_path, "wb"))
            
            return execution_feedback_str, execution_model_output
        
        except Exception as e:
            return f"Execution error: {e}", None


ModelExperiment = Experiment
