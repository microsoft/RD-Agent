from rdagent.core.task_generator import TaskGenerator
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment


class QlibFactorRunner(TaskGenerator[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`

    - TODO: implement a qlib handler
    """

    def generate(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        return exp  # TODO IMPLEMENT THIS
