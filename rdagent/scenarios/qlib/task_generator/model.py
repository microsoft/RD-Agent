from rdagent.components.coder.model_coder.model import ModelExperiment, ModelImplementation
from rdagent.utils.env import QTDockerEnv
import shutil
from pathlib import Path
import uuid
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
    def generate(self, exp: ModelExperiment) -> ModelExperiment:
        TEMPLATE_PATH = Path("/home/v-xisenwang/RD-Agent/test/utils/env_tpl")  #Can be updated 

        # To prepare
        unique_id = uuid.uuid4()
        self.workspace_path = Path("/home/v-xisenwang/RD-Agent/test") / f"M{unique_id}"  # need to set base workspace path
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # to copy_template_to_workspace
        for file_name in ["model.py", "read_exp.py", "conf.yaml"]:
            shutil.copyfile(TEMPLATE_PATH / file_name, self.workspace_path / file_name)

        # Assign it to exp.workspace's varaible 

        # to replace & inject code
        code_implementation = exp.sub_implementations[0].code 

        exp.sub_implementations[0].inject_code(**{"model.py": code_implementation})

        if exp.sub_tasks[0].model_type == "TimesSeries":
            env_to_use = {
            "dataset_cls": "TSDatasetH",
            "step_len": 20
            }

        if exp.sub_tasks[0].model_type == "Tabular":
            env_to_use = {
            "dataset_cls": "DatasetH"
            }

        # to execute
        qtde = QTDockerEnv()
        
        # Preparing the Docker environment
        qtde.prepare()
        
        # Run the Docker container with the specified entry
        
        result = qtde.run(local_path=self.workspace_path, entry="qrun conf.yaml", env={{"PYTHONPATH": "./"}, env_to_use})
        print(result)
        
        # Run the experiment analysis code
        result = qtde.run(local_path=self.workspace_path, entry="python read_exp.py")
        print(result)

        exp.result = result 

        return exp  

    

    # # def __init__(self, **kwargs):
    # #     self.kwargs = kwargs
    # #     self.prepare()

    # def prepare(self) -> None:
    #     """
    #     Prepare for the workspace;
    #     """
        

    # def copy_template_to_workspace(self):
    #     """
    #     Copy the template files to the new workspace.
    #     """
    #     for file_name in ["model.py", "read_exp.py", "conf.yaml"]:
    #         shutil.copyfile(self.TEMPLATE_PATH / file_name, self.workspace_path / file_name)

    # def execute(self):
    #     """
    #     Execute the Qlib model in the prepared workspace.
    #     """
    #     qtde = QTDockerEnv()
        
    #     # Preparing the Docker environment
    #     qtde.prepare()
        
    #     # Run the Docker container with the specified entry
    #     result = qtde.run(local_path=self.workspace_path, entry="qrun conf.yaml", env={"PYTHONPATH": "./"})
    #     print(result)
        
    #     # Run the experiment analysis code
    #     result = qtde.run(local_path=self.workspace_path, entry="python read_exp.py")
    #     print(result)

    # def clearFile(self):
    #     qtde = QTDockerEnv()
        
    #     # Preparing the Docker environment
    #     qtde.prepare()
        
    #     # Run the Docker container with the specified entry
    #     result = qtde.run(local_path=self.workspace_path, entry="rm -r RD-Agent/test/Mbcee0d94-5e05-4c55-991b-4d64332edd93.yaml", env={"PYTHONPATH": "./"})
    #     print(result)


# Example usage
# sampleImp = QlibModelRunner()
# sampleImp.clearFile()

# sampleImp = QlibModelImplementation()
# sampleImp.execute()
