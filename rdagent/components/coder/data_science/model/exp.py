import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.env import DockerEnv, DSDockerConf
# TODO: Complete the implementation of the class DataLoaderTask and class DataLoaderFBWorkspace


class ModelTask(CoSTEERTask):
    def __init__(
        self,
        name: str,
        description: str,
        architecture: str,
        *args,
        hyperparameters: Dict[str, str],
        formulation: str = None,
        variables: Dict[str, str] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.formulation: str = formulation
        self.architecture: str = architecture
        self.variables: str = variables
        self.hyperparameters: str = hyperparameters
        self.model_type: str = (
            model_type  # Tabular for tabular model, TimesSeries for time series model, Graph for graph model, XGBoost for XGBoost model
        )
        super().__init__(name=name, description=description, *args, **kwargs)

    def get_task_information(self):
        task_desc = f"""name: {self.name}
description: {self.description}
"""
        task_desc += f"formulation: {self.formulation}\n" if self.formulation else ""
        task_desc += f"architecture: {self.architecture}\n"
        task_desc += f"variables: {self.variables}\n" if self.variables else ""
        task_desc += f"hyperparameters: {self.hyperparameters}\n"
        task_desc += f"model_type: {self.model_type}\n"
        return task_desc

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

    We support two ways of interface:
        (version 1) for qlib we'll make a script to import the model in the implementation in file `model.py` after setting the cwd into the directory
            - from model import model_cls
            - initialize the model by initializing it `model_cls(input_dim=INPUT_DIM)`
            - And then verify the model.

        (version 2) for kaggle we'll make a script to call the fit and predict function in the implementation in file `model.py` after setting the cwd into the directory
    """

    def hash_func(
        self,
        batch_size: int = 8,
        num_features: int = 10,
        num_timesteps: int = 4,
        num_edges: int = 20,
        input_value: float = 1.0,
        param_init_value: float = 1.0,
    ) -> str:
        target_file_name = f"{batch_size}_{num_features}_{num_timesteps}_{input_value}_{param_init_value}"
        for code_file_name in sorted(list(self.code_dict.keys())):
            target_file_name = f"{target_file_name}_{self.code_dict[code_file_name]}"
        return md5_hash(target_file_name)

    @cache_with_pickle(hash_func)
    def execute(
        self
    ):
        super().execute()
        try:
            de = DockerEnv(DSDockerConf())
            de.prepare()

            dump_code = (Path(__file__).parent / "model_execute_template.txt").read_text()

            log, results = de.dump_python_code_run_and_get_results(
                code=dump_code,
                dump_file_names=["execution_feedback_str.pkl", "execution_model_output.pkl"],
                local_path=str(self.workspace_path),
                env={},
                code_dump_file_py_name="model_test",
            )
            if results is None:
                raise RuntimeError(f"Error in running the model code: {log}")
            [execution_feedback_str, execution_model_output] = results

        except Exception as e:
            execution_feedback_str = f"Execution error: {e}\nTraceback: {traceback.format_exc()}"
            execution_model_output = None

        if len(execution_feedback_str) > 2000:
            execution_feedback_str = (
                execution_feedback_str[:1000] + "....hidden long error message...." + execution_feedback_str[-1000:]
            )
        return execution_feedback_str, execution_model_output