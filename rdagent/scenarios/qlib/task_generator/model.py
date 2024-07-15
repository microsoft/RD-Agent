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
        TEMPLATE_PATH = Path("RD-Agent/test/utils/model_template")  #Can be updated 

        # To prepare
        unique_id = uuid.uuid4()
        self.workspace_path = Path("RD-Agent/test/testOutputs") / f"M{unique_id}"  # need to set base workspace path
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # to copy_template_to_workspace
        for file_name in ["model.py", "read_exp.py", "conf.yaml"]:
            shutil.copyfile(TEMPLATE_PATH / file_name, self.workspace_path / file_name)

        # Assign it to exp.workspace's varaible 

        # to replace & inject code
        code_implementation = exp.sub_implementations[0].code_dict['model.py']

        exp.sub_implementations[0].inject_code(**{"model.py": code_implementation})

        # Write the code implementation directly to the model.py file
        with open(self.workspace_path / "model.py", "w") as f:
            f.write(code_implementation)

        env_to_use = {}

        if exp.sub_tasks[0].model_type == "TimeSeries":
            env_to_use = {
            "dataset_cls": "TSDatasetH",
            "step_len": 20,
            "num_timesteps": 20
            }

        if exp.sub_tasks[0].model_type == "Tabular":
            env_to_use = {
            "dataset_cls": "DatasetH"
            }

        print("Model Type is:", exp.sub_tasks[0].model_type)

        # to execute
        qtde = QTDockerEnv()
        
        # Preparing the Docker environment
        qtde.prepare()
        
        # Run the Docker container with the specified entry
        
        result = qtde.run(local_path=self.workspace_path, entry="qrun conf.yaml", env={"PYTHONPATH": "./", **env_to_use})
        print(result)
        
        # Run the experiment analysis code
        result = qtde.run(local_path=self.workspace_path, entry="python read_exp.py")
        print(result)

        exp.result = result 

        return exp  

