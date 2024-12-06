import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, DSDockerConf

# TODO: Complete the implementation of the class DataLoaderTask and class DataLoaderFBWorkspace


class DataLoaderTask(CoSTEERTask):
    def __init__(
        self,
        name: str,
        description: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name=name, description=description, *args, **kwargs)

    def get_task_information(self):
        task_desc = f"""name: {self.name}
description: {self.description}
"""
        return task_desc

    @staticmethod
    def from_dict(dict):
        return DataLoaderTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


class DataLoaderFBWorkspace(FBWorkspace):

    # TODO: use the cache_with_pickle decorator.
    def execute(self):
        super().execute()
        try:
            de = DockerEnv(conf=DSDockerConf())
            de.prepare()

            # TODO: UNIT TEST for data loader
            dump_code = T(".prompts:data_loader_execute_code").r()
            log, results = de.dump_python_code_run_and_get_results(
                code=dump_code,
                dump_file_names=["data.pkl"],
                local_path=str(self.workspace_path),
                code_dump_file_py_name="execute_data_loader",
            )

            if results is None:
                raise RuntimeError(f"Failed to execute load_data.py, Log: {log}")
            # TODO: Cache the processed data into a pickle file
            execution_feedback = "Execution successful"
            preprocessed_data = results[0]

        except Exception as e:
            execution_feedback = f"Execution error: {e}\nTraceback: {traceback.format_exc()}"
            preprocessed_data = None

        return execution_feedback, preprocessed_data
