"""FT Trace - Specialized Trace for LLM Fine-tuning scenario.

Provides SOTA experiment tracking functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rdagent.components.coder.finetune.conf import (
    FT_DATA_SCRIPT_NAME,
    FT_YAML_FILE_NAME,
)
from rdagent.core.evolving_framework import KnowledgeBase
from rdagent.core.proposal import ExperimentFeedback, Trace
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
    from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen


class FTTrace(Trace["LLMFinetuneScen", KnowledgeBase]):
    """Specialized Trace for LLM Fine-tuning scenario.

    Adds SOTA experiment tracking on top of the base Trace class.
    SOTA is explicitly managed via DAG traversal.
    """

    def __init__(self, scen: "LLMFinetuneScen", knowledge_base: KnowledgeBase | None = None) -> None:
        super().__init__(scen, knowledge_base)

        # Type hint for linting
        self.hist: list[tuple[FTExperiment, ExperimentFeedback]] = []

    def sota_benchmark(self) -> dict | None:
        """Return SOTA experiment's benchmark results."""
        sota_exp = self.get_sota_experiment()
        if sota_exp is None:
            return None
        ws = sota_exp.experiment_workspace
        if ws is None or ws.running_info is None:
            return None
        result = getattr(ws.running_info, "result", None)
        if result and isinstance(result, dict) and "benchmark" in result:
            return result["benchmark"]
        return None

    def get_experiment_info(self, exp: "FTExperiment") -> dict[str, Any]:
        """Return experiment's full info for hypothesis generation."""
        info: dict[str, Any] = {
            "hypothesis": str(exp.hypothesis) if exp.hypothesis else None,
            "config": None,
            "benchmark": None,
            "data_script": None,
        }

        ws = exp.experiment_workspace
        if ws is None:
            return info

        if ws.file_dict:
            if FT_YAML_FILE_NAME in ws.file_dict:
                info["config"] = ws.file_dict[FT_YAML_FILE_NAME]
            if FT_DATA_SCRIPT_NAME in ws.file_dict:
                info["data_script"] = ws.file_dict[FT_DATA_SCRIPT_NAME]

        if ws.running_info:
            result = getattr(ws.running_info, "result", None)
            if result and isinstance(result, dict) and "benchmark" in result:
                info["benchmark"] = result["benchmark"].get("accuracy_summary")

        return info

    def sota_info(self) -> dict[str, Any] | None:
        """Return SOTA experiment's full info for hypothesis generation."""
        sota_exp = self.get_sota_experiment()
        if sota_exp is None:
            return None
        return self.get_experiment_info(sota_exp)
