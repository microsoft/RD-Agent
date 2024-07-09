import json
import uuid
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.model_coder.conf import MODEL_IMPL_SETTINGS
from rdagent.core.exception import CodeFormatException
from rdagent.core.experiment import Experiment, FBImplementation, Task
from rdagent.utils import get_module_by_module_path


class ModelTask(Task):
    # TODO: it should change when the Task changes.
    name: str
    description: str
    formulation: str
    variables: Dict[str, str]  # map the variable name to the variable description

    def __init__(
        self, name: str, description: str, formulation: str, variables: Dict[str, str], key: Optional[str] = None
    ) -> None:
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

    def get_information(self):
        return f"""name: {self.name}
description: {self.description}
formulation: {self.formulation}
variables: {self.variables}
key: {self.key}
"""

    @staticmethod
    def from_dict(dict):
        return ModelTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


class ModelImplementation(FBImplementation):
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

    def __init__(self, target_task: Task) -> None:
        super().__init__(target_task)

    def prepare(self) -> None:
        """
        Prepare for the workspace;
        """
        unique_id = uuid.uuid4()
        self.workspace_path = MODEL_IMPL_SETTINGS.workspace_path / f"M{unique_id}"
        # start with `M` so that it can be imported via python
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    def execute(self, data=None, config: dict = {}):
        mod = get_module_by_module_path(str(self.workspace_path / "model.py"))
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
  - So your implemented code could follow the following pattern
    ```Python
    class XXXLayer(torch.nn.Module):
        ...
    model_cls = XXXLayer
    ```
- initialize the model by initializing it `model_cls(input_dim=INPUT_DIM)`
- And then verify the model by comparing the output tensors by feeding specific input tensor.
"""


class ModelExperiment(Experiment[ModelTask, ModelImplementation]): ...
