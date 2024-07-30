from pathlib import Path

from rdagent.components.coder.model_coder.model import ModelExperiment
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
# General Model Scenario

## Overview

This demo automates the extraction and iterative development of models from academic papers, ensuring functionality and correctness.

### Scenario: Auto-Developing Model Code from Academic Papers

#### Overview

This scenario automates the development of PyTorch models by reading academic papers or other sources. It supports various data types, including tabular, time-series, and graph data. The primary workflow involves two main components: the Reader and the Coder.

#### Workflow Components

1. **Reader**
    - Parses and extracts relevant model information from academic papers or sources, including architectures, parameters, and implementation details.
    - Uses Large Language Models to convert content into a structured format for the Coder.

2. **Evolving Coder**
    - Translates structured information from the Reader into executable PyTorch code.
    - Utilizes an evolving coding mechanism to ensure correct tensor shapes, verified with sample input tensors.
    - Iteratively refines the code to align with source material specifications.

#### Supported Data Types

- **Tabular Data:** Structured data with rows and columns, such as spreadsheets or databases.
- **Time-Series Data:** Sequential data points indexed in time order, useful for forecasting and temporal pattern recognition.
- **Graph Data:** Data structured as nodes and edges, suitable for network analysis and relational tasks.

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
