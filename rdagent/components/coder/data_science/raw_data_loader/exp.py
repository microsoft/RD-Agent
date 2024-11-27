import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.env import KGDockerEnv, QTDockerEnv

# TODO: Complete the implementation of the class DataLoaderTask and class DataLoaderFBWorkspace


class DataLoaderTask(CoSTEERTask):
    def __init__(
        self,
        name: str,
        description: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name=name, desc=description, *args, **kwargs)

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
    def execute(self):
        super().execute()
        try:
            qtde = QTDockerEnv() if self.target_task.version == 1 else KGDockerEnv()
            qtde.prepare()

            # TODO: UNIT TEST for data loader
            dump_code = (Path(__file__).parent / "data_loader_unit_test.txt").read_text()

            # TODO: Cache the processed data into a pickle file
            pass

        except Exception as e:
            pass


DataLoaderExperiment = Experiment
