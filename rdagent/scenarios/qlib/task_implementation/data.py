from rdagent.core.task import FBTaskImplementation


class QlibDataTaskImplementation(FBTaskImplementation):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`

    - TODO: implement a qlib handler
    """
