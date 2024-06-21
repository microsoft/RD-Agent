import torch
from pathlib import Path
import uuid
from typing import Dict, Optional, Sequence
from rdagent.core.exception import CodeFormatException
from rdagent.core.task import BaseTask, FBTaskImplementation, ImpLoader, TaskImplementation, TaskLoader
from rdagent.model_implementation.conf import MODEL_IMPL_SETTINGS
from rdagent.utils import get_module_by_module_path


class ModelImplTask(BaseTask):
    # TODO: it should change when the BaseTask changes.
    name: str
    description: str
    formulation: str
    variables: Dict[str, str]  # map the variable name to the variable description

    def __init__(self, name: str, description: str, formulation: str, variables: Dict[str, str], key: Optional[str] = None) -> None:
        """

        Parameters
        ----------

        key : Optional[str]
            Key is a string to identify the task.
            It will be used to connect to other information(e.g. ground truth).
        """
        self.name = name
        self.description = description
        self.formulation = formulation
        self.variables = variables
        self.key = key


class ModelTaskLoderJson(TaskLoader):
    def __init__(self, json_uri: str) -> None:
        super().__init__()
        # TODO: the json should be loaded from URI.
        self.json_uri = json_uri

    def load(self, *argT, **kwargs) -> Sequence[ModelImplTask]:
        # TODO: we should load the tasks from json;

        formula_info = {
            "name": "Anti-Symmetric Deep Graph Network (A-DGN)",
            "description": "A framework for stable and non-dissipative DGN design. It ensures long-range information preservation between nodes and prevents gradient vanishing or explosion during training.",
            "formulation": "x_u^{(l)} = x_u^{(l-1)} + \\epsilon \\sigma \\left( W^T x_u^{(l-1)} + \\Phi(X^{(l-1)}, N_u) + b \\right)",
            "variables": {
                "x_u^{(l)}": "The state of node u at layer l",
                "\\epsilon": "The step size in the Euler discretization",
                "\\sigma": "A monotonically non-decreasing activation function",
                "W": "An anti-symmetric weight matrix",
                "X^{(l-1)}": "The node feature matrix at layer l-1",
                "N_u": "The set of neighbors of node u",
                "b": "A bias vector",
            },
            "key": "A-DGN",
        }
        return [ModelImplTask(**formula_info)]


class ModelTaskImpl(FBTaskImplementation):
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
    - And then verify the modle.

    """
    def __init__(self, target_task: BaseTask) -> None:
        super().__init__(target_task)
        self.path = None

    def prepare(self) -> None:
        """
        Prepare for the workspace;
        """
        unique_id = uuid.uuid4()
        self.path = MODEL_IMPL_SETTINGS.workspace_path / f"M{unique_id}"
        # start with `M` so that it can be imported via python
        self.path.mkdir(parents=True, exist_ok=True)

    def execute(self, data=None, config: dict = {}):
        mod = get_module_by_module_path(str(self.path / "model.py"))
        try:
            model_cls = mod.model_cls
        except AttributeError:
            raise CodeFormatException("The model_cls is not implemented in the model.py")
        # model_init = 

        assert isinstance(data, tuple)
        node_feature, _ = data
        in_channels = node_feature.size(-1)
        m = model_cls(in_channels)

        # TODO: initialize all the parameters of `m` to `model_eval_param_init`
        model_eval_param_init: float = config["model_eval_param_init"]

        # initialize all parameters of `m` to `model_eval_param_init`
        for _, param in m.named_parameters():
            param.data.fill_(model_eval_param_init)

        assert isinstance(data, tuple)
        return m(*data)

    def execute_desc(self) -> str:
        return """
The the implemented code will be placed in a file like <uuid>/model.py

We'll import the model in the implementation in file `model.py` after setting the cwd into the directory
- from model import model_cls (So you must have a variable named `model_cls` in the file)
  - So your implelemented code could follow the following pattern
    ```Python
    class XXXLayer(torch.nn.Module):
        ...
    model_cls = XXXLayer
    ```
- initialize the model by initializing it `model_cls(input_dim=INPUT_DIM)`
- And then verify the model by comparing the output tensors by feeding specific input tensor.
"""

class ModelImpLoader(ImpLoader[ModelImplTask, ModelTaskImpl]):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self, task: ModelImplTask) -> ModelTaskImpl:
        assert task.key is not None
        mti = ModelTaskImpl(task)
        mti.prepare()
        with open(self.path / f"{task.key}.py", "r") as f:
            code = f.read()
        mti.inject_code(**{"model.py": code})
        return mti
