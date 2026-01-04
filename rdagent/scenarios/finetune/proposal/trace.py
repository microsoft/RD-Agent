"""FT Trace - Specialized Trace for LLM Fine-tuning scenario.

Provides SOTA experiment tracking functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rdagent.components.coder.finetune.conf import FT_DATA_FILE_NAME, FT_DATA_SCRIPT_NAME, FT_YAML_FILE_NAME
from rdagent.core.evolving_framework import KnowledgeBase
from rdagent.core.proposal import ExperimentFeedback, Trace
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
    from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen


class FTTrace(Trace["LLMFinetuneScen", KnowledgeBase]):
    """Specialized Trace for LLM Fine-tuning scenario.

    Adds SOTA experiment tracking on top of the base Trace class.
    SOTA is explicitly managed via update_sota() based on LLM judgment.
    """

    def __init__(self, scen: "LLMFinetuneScen", knowledge_base: KnowledgeBase | None = None) -> None:
        super().__init__(scen, knowledge_base)

        # Type hint for linting
        self.hist: list[tuple[FTExperiment, ExperimentFeedback]] = []

        # Explicitly stored SOTA experiment (updated by LLM judgment in feedback)
        self._sota_exp: "FTExperiment | None" = None

    def sota_experiment(self) -> "FTExperiment | None":
        """Return the current SOTA experiment."""
        return self._sota_exp

    def update_sota(self, exp: "FTExperiment") -> None:
        """Update the SOTA experiment based on LLM judgment.

        Also syncs file_dict from disk to ensure consistency after CoSTEER's
        recover_ws_ckp() which only restores files but doesn't update file_dict.
        """
        ws = exp.experiment_workspace
        if ws is not None and ws.workspace_path.exists():
            # Sync file_dict from disk for key config files
            # This ensures sota_info() and inject_code_from_file_dict() use correct content
            white_list = [FT_YAML_FILE_NAME, FT_DATA_FILE_NAME, "dataset_info.json"]
            for file_path in ws.workspace_path.rglob("*"):
                if file_path.is_file() and file_path.name in white_list:
                    relative = str(file_path.relative_to(ws.workspace_path))
                    try:
                        ws.file_dict[relative] = file_path.read_text()
                    except UnicodeDecodeError:
                        pass  # Skip binary files

        self._sota_exp = exp
        loop_id = self.idx2loop_id.get(len(self.hist), "?")
        logger.info(f"SOTA updated to experiment from loop {loop_id}")

    def sota_benchmark(self) -> dict | None:
        """Return SOTA experiment's benchmark results."""
        sota_exp = self.sota_experiment()
        if sota_exp is None:
            return None
        ws = sota_exp.experiment_workspace
        if ws is None or ws.running_info is None:
            return None
        result = getattr(ws.running_info, "result", None)
        if result and isinstance(result, dict) and "benchmark" in result:
            return result["benchmark"]
        return None

    def sota_info(self) -> dict[str, Any] | None:
        """Return SOTA experiment's full info for hypothesis generation."""
        sota_exp = self.sota_experiment()
        if sota_exp is None:
            return None

        info: dict[str, Any] = {
            "hypothesis": str(sota_exp.hypothesis) if sota_exp.hypothesis else None,
            "config": None,
            "benchmark": None,
            "data_script": None,
        }

        ws = sota_exp.experiment_workspace
        if ws is None:
            return info

        if ws.file_dict:
            if "train.yaml" in ws.file_dict:
                info["config"] = ws.file_dict["train.yaml"]
            if FT_DATA_SCRIPT_NAME in ws.file_dict:
                info["data_script"] = ws.file_dict[FT_DATA_SCRIPT_NAME]

        if ws.running_info:
            result = getattr(ws.running_info, "result", None)
            if result and isinstance(result, dict) and "benchmark" in result:
                info["benchmark"] = result["benchmark"].get("accuracy_summary")

        return info
