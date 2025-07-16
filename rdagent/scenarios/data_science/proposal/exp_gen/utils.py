from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.utils.agent.tpl import T

_COMPONENT_META: Dict[str, Dict[str, Any]] = {
    "DataLoadSpec": {
        "target_name": "Data loader and specification generation",
        "spec_file": "spec/data_loader.md",
        "output_format_key": ".prompts:output_format.data_loader",
        "task_class": DataLoaderTask,
    },
    "FeatureEng": {
        "target_name": "Feature engineering",
        "spec_file": "spec/feature.md",
        "output_format_key": ".prompts:output_format.feature",
        "task_class": FeatureTask,
    },
    "Model": {
        "target_name": "Model",
        "spec_file": "spec/model.md",
        "output_format_key": ".prompts:output_format.model",
        "task_class": ModelTask,
    },
    "Ensemble": {
        "target_name": "Ensemble",
        "spec_file": "spec/ensemble.md",
        "output_format_key": ".prompts:output_format.ensemble",
        "task_class": EnsembleTask,
    },
    "Workflow": {
        "target_name": "Workflow",
        "spec_file": "spec/workflow.md",
        "output_format_key": ".prompts:output_format.workflow",
        "task_class": WorkflowTask,
    },
    "Pipeline": {
        "target_name": "Pipeline",
        "spec_file": None,
        "output_format_key": ".prompts:output_format.pipeline",
        "task_class": PipelineTask,
    },
}


def get_component(name: str) -> Dict[str, Any]:
    meta = _COMPONENT_META.get(name)
    if meta is None:
        raise KeyError(f"Unknown component: {name!r}")

    return {
        "target_name": meta["target_name"],
        "spec_file": meta["spec_file"],
        "task_output_format": T(meta["output_format_key"]).r(),
        "task_class": meta["task_class"],
    }


class CodingSketch(BaseModel):
    current_state: str = Field(
        description="A summary of the current `main.py` script that serves as the baseline for the planned changes. Focusing on parts that are related to the hypothesis. If `main.py` does not yet exist (i.e., it will be created from scratch based on this sketch), use the string 'N/A'."
    )
    modifications: List[str] = Field(
        description="A list of specific, targeted changes to be applied to the existing code identified in `current_state`. Each string in the list should concisely describe (in 3-4 sentences): "
        "(a) the specific part of the code to be altered (e.g., a function name, a class, or a logical block); "
        "(b) the nature of the modification (e.g., bug fix, feature addition, refactoring of a small section, performance optimization, deletion); and "
        "(c) a brief explanation or high-level sketch of the new logic or change. "
        "If no direct modifications to existing code are planned (e.g., if creating an entirely new `main.py` as detailed in `structure`), this list should be empty."
    )
    structure: List[str] = Field(
        description="An outline of the new high-level architectural components (primarily functions and classes) if a new `main.py` script is being created from scratch, or if the existing `main.py` is undergoing a major refactor that fundamentally alters or replaces its core structure. "
        "Each string in the list should define a planned function or class, detailing its name, primary responsibility, key parameters (if applicable), return values (if applicable), and core functionality in 2-3 sentences. "
        "This field is typically used when `current_state` is 'N/A' or when the scope of change requires a new architectural blueprint rather than just targeted `modifications`. "
        "Leave empty if the plan only involves direct `modifications` to the existing structure in `current_state`."
    )
    sketch: str = Field(
        description="A detailed, step-by-step narrative that elaborates on how to implement the planned code. "
        "This section should synthesize the information from `modifications` (if any) and/or `structure` (if any) into a comprehensive and actionable coding plan for `main.py`. "
        "The content **must** be formatted using Markdown, with logical sections, key decision points, or implementation steps clearly organized by level-3 headings (i.e., `###`). "
        "This field should provide sufficient detail for a developer to understand the implementation flow, algorithms, data handling, and key logic points without ambiguity."
    )
