from rdagent.components.coder.model_coder.model import ModelImplementation
from rdagent.core.task_generator import TaskGenerator
from rdagent.utils.env import QTDockerEnv
import shutil
from pathlib import Path

class QlibModelImplementation(TaskGenerator[ModelImplementation]):
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

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.local_path = kwargs.get('local_path', str(Path(__file__).resolve().parent / "env_tpl"))

    def execute(self):
        """
        docker run  -v <path to model>:<model in the image> qlib_image
        qpte = QTDockerEnv
        qpte.run(local_path=self.XXXXpath, entry="qrun ...")
        # TODO: inject exp analysis code
        qpte.run(local_path=self.XXXXpath, entry="python read_exp.py")
        append following code into read_exp.py
        to_pickle() or to_csv()   # which does not depend on environment.
        read_csv or pickle in RD-Agent Env;
        """
        qtde = QTDockerEnv()
        
        # Preparing the Docker environment
        qtde.prepare()
        
        # Inject experiment analysis code from the provided path
        self.inject_exp_analysis_code(self.local_path)
        
        # Run the Docker container with the specified entry
        result = qtde.run(local_path=self.local_path, entry="qrun conf.yaml", env={"PYTHONPATH": "./"})
        print(result)
        
        # Run the experiment analysis code
        result = qtde.run(local_path=self.local_path, entry="python read_exp.py")
        print(result)

    def inject_exp_analysis_code(self, local_path):
        source_path = '/home/v-xisenwang/RD-Agent/test/utils/env_tpl/read_exp.py'
        destination_path = Path(local_path) / 'read_exp.py'
        shutil.copyfile(source_path, destination_path)

# sampleImp = QlibModelImplementation()
# sampleImp.execute()