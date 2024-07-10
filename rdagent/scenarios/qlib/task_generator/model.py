from rdagent.components.coder.model_coder.model import ModelImplementation
from rdagent.core.task_generator import TaskGenerator


class QlibModelRunner(TaskGenerator[ModelImplementation]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - Pytorch `model.py`
    - results in `mlflow`

    https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_nn.py
    - pt_model_uri:  hard-code `model.py:Net` in the config
    - let LLM modify model.py
    """

    def generate(self, exp: ModelImplementation) -> ModelImplementation:
        return exp  # TODO IMPLEMENT THIS
