from copy import deepcopy
from pathlib import Path

from rdagent.core.experiment import Task
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro

# Factor
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)

# Model
from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)


prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class QlibFactorExperiment(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")

class QlibModelExperiment(ModelExperiment[ModelTask, QlibFBWorkspace, ModelFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "model_template")

class QlibQuantScenario(Scenario):
    def __init__(self) -> None:
        super().__init__()
        self._source_data = deepcopy(get_data_folder_intro())

        self._rich_style_description = deepcopy(prompt_dict["qlib_factor_rich_style_description"])
        self._experiment_setting = deepcopy(prompt_dict["qlib_factor_experiment_setting"])

    def background(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        quant_background = "The background of the scenario is as follows:\n" + prompt_dict["qlib_quant_background"]
        factor_background = "This time, I need your help with the research and development of the factor. The background of the factor scenario is as follows:\n" + prompt_dict["qlib_factor_background"]
        model_background = "This time, I need your help with the research and development of the model. The background of the model scenario is as follows:\n" + prompt_dict["qlib_model_background"]

        # TODO: There are some issues here
        if tag is None:
            return quant_background + "\n" + factor_background + "\n" + model_background
        elif tag == "factor":
            return factor_background
        else:
            return model_background

    def get_source_data_desc(self) -> str:
        return self._source_data

    def output_format(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        factor_output_format = "The factor code should output the following format:\n" + prompt_dict["qlib_factor_output_format"]
        model_output_format = "The model code should output the following format:\n" + prompt_dict["qlib_model_output_format"]

        if tag is None:
            return factor_output_format + "\n" + model_output_format
        elif tag == "factor":
            return factor_output_format
        else:
            return model_output_format

    def interface(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        factor_interface = "The factor code should be written in the following interface:\n" + prompt_dict["qlib_factor_interface"]
        model_interface = "The model code should be written in the following interface:\n" + prompt_dict["qlib_model_interface"]

        if tag is None:
            return factor_interface + "\n" + model_interface
        elif tag == "factor":
            return factor_interface
        else:
            return model_interface

    def simulator(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        factor_simulator = "The factor code will be sent to the simulator:\n" + prompt_dict["qlib_factor_simulator"]
        model_simulator = "The model code will be sent to the simulator:\n" + prompt_dict["qlib_model_simulator"]

        if tag is None:
            return factor_simulator + "\n" + model_simulator
        elif tag == "factor":
            return factor_simulator
        else:
            return model_simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        def common_description() -> str:
            return f"""\n------Background of the scenario------
{self.background}

------The source dataset you can use------
{self.get_source_data_desc()}
"""
        # TODO: There are still some issues with handling source_data here
        def source_data() -> str:
            return f"""
------The source data you can use------
{self.get_source_data_desc()}
"""
        
        def interface(tag: str | None) -> str:
            return f"""
------The interface you should follow to write the runnable code------
{self.interface(tag)}
"""
        
        def output(tag: str | None) -> str:
            return f"""
------The output of your code should be in the format------
{self.output_format(tag)}
"""
        
        def simulator(tag: str | None) -> str:
            return f"""
------The simulator user can use to test your solution------
{self.simulator(tag)}
"""
        if simple_background:
            return common_description() 
        elif filtered_tag == "hypothesis_and_experiment" or filtered_tag == "feedback":
            return common_description() + simulator(None)
        elif filtered_tag == "factor" or filtered_tag == "feature":
            return common_description() + interface("factor") + output("factor") + simulator("factor")
        else:
            return common_description() + interface("model") + output("model") + simulator("model")



#         """A static scenario describer"""
#         if simple_background:
#             return f"""Background of the scenario:
# {self.background}"""
#         return f"""Background of the scenario:
# {self.background}
# The source data you can use:
# {self.get_source_data_desc(task)}
# The interface you should follow to write the runnable code:
# {self.interface}
# The output of your code should be in the format:
# {self.output_format}
# The simulator user can use to test your factor:
# {self.simulator}
# """
