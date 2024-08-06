from pathlib import Path

from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class GeneralModelScenario(Scenario):
    @property
    def background(self) -> str:
        return prompt_dict["general_model_background"]

    @property
    def source_data(self) -> str:
        raise NotImplementedError("source_data of GeneralModelScenario is not implemented")

    @property
    def output_format(self) -> str:
        return prompt_dict["general_model_output_format"]

    @property
    def interface(self) -> str:
        return prompt_dict["general_model_interface"]

    @property
    def simulator(self) -> str:
        return prompt_dict["general_model_simulator"]

    @property
    def rich_style_description(self) -> str:
        return """
### [Model Research & Development Co-Pilot](#_scenario)

#### [Overview](#_summary)

This demo automates the extraction and development of PyTorch models from academic papers. It supports various model types through two main components: Reader and Coder.
 
#### [Workflow Components](#_rdloops)
 
1. **[Reader](#_research)**
    - Extracts model information from papers, including architectures and parameters.
    - Converts content into a structured format using Large Language Models.
 
2. **[Evolving Coder](#_development)**
    - Translates structured information into executable PyTorch code.
    - Ensures correct tensor shapes with an evolving coding mechanism.
    - Refines the code to match source specifications.

        """

    def get_scenario_all_desc(self) -> str:
        return f"""Background of the scenario:
{self.background}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your model:
{self.simulator}
"""
