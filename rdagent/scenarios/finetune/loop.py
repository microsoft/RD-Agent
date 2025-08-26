from pathlib import Path
from typing import Any

from rdagent.app.finetune.llm.conf import LLMFinetunePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.data_process.data_format_converter import (
    DataFormatConverter,
)


class LLMFinetuneRDLoop(RDLoop):
    """LLM fine-tuning loop using standard RDLoop workflow"""

    def __init__(self, PROP_SETTING: LLMFinetunePropSetting):
        # Initialize scenario first
        scen = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")

        # Initialize RDLoop components
        self.hypothesis_gen = import_class(PROP_SETTING.hypothesis_gen)(scen)
        self.hypothesis2experiment = import_class(PROP_SETTING.hypothesis2experiment)()
        self.coder = import_class(PROP_SETTING.coder)(scen)
        self.runner = import_class(PROP_SETTING.runner)(scen)
        self.summarizer = import_class(PROP_SETTING.summarizer)(scen)

        # Initialize trace
        from rdagent.core.proposal import Trace

        self.trace = Trace(scen=scen)

        # Store finetune settings
        self.ft_rd_setting = PROP_SETTING
        self.dataset = PROP_SETTING.dataset
        self.model = PROP_SETTING.base_model_name

        # Setup environment
        self._setup_environment()

        # Preprocess data during initialization
        self._preprocess_data()

        # Initialize LoopBase (skip RDLoop.__init__ to avoid double initialization)
        from rdagent.utils.workflow import LoopBase

        LoopBase.__init__(self)

    def _setup_environment(self):
        """Setup Docker environment with proper volume mappings"""
        data_volumes = {}

        local_path = self.ft_rd_setting.file_path
        data_volumes[local_path] = {
            "bind": "/data",
            "mode": "ro",
        }

        # Create base directories
        finetune_base_dir = Path(local_path)
        finetune_base_dir.mkdir(parents=True, exist_ok=True)
        (finetune_base_dir / "output").mkdir(parents=True, exist_ok=True)

        from rdagent.components.coder.finetune.conf import get_ft_env

        self.env = get_ft_env(
            extra_volumes=data_volumes,
            running_timeout_period=None,
            enable_cache=False,
        )

    def _preprocess_data(self):
        """Preprocess dataset format during initialization"""
        # Use dataset-specific preprocessed data directory
        finetune_base_dir = Path(self.ft_rd_setting.file_path)
        preprocessed_dir = finetune_base_dir / "preprocessed_data" / self.dataset

        # Check if preprocessed data already exists
        expected_files = ["processed_dataset.json", "dataset_info.json"]
        if all((preprocessed_dir / file_name).exists() for file_name in expected_files):
            logger.info(f"Preprocessed data already exists at {preprocessed_dir}, skipping preprocessing")
            return

        logger.info("Preprocessing dataset format...")
        # Create preprocessed directory
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        data_converter = DataFormatConverter(
            dataset=self.dataset,
            model=self.model,
            ft_rd_setting=self.ft_rd_setting,
            scen=import_class(self.ft_rd_setting.scen)(),
        )

        success = data_converter.convert_dataset(self.env, preprocessed_dir)
        if not success:
            raise RuntimeError("Failed to preprocess dataset")

        logger.info("Dataset preprocessing completed")

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        """Generate LLM fine-tuning experiment"""
        if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
            exp = self.hypothesis_gen.gen(self.trace)
            logger.log_object(exp.sub_tasks, tag="experiment generation")
            return exp

        import asyncio

        await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        """Generate fine-tuning code"""
        exp = prev_out["direct_exp_gen"]

        # Convert pending_tasks_list to sub_tasks like in data_science loop
        if hasattr(exp, "pending_tasks_list") and exp.pending_tasks_list:
            for tasks in exp.pending_tasks_list:
                exp.sub_tasks = tasks
                break  # For finetune, we typically have only one task group

        exp = self.coder.develop(exp)
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        """Execute fine-tuning experiment"""
        exp = prev_out["coding"]
        exp = self.runner.develop(exp)
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        """Generate feedback from experiment results"""
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            from rdagent.core.proposal import HypothesisFeedback

            feedback = HypothesisFeedback(
                observations=str(e),
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=False,
            )
            logger.log_object(feedback, tag="feedback")
            self.trace.hist.append((prev_out["direct_exp_gen"], feedback))
        else:
            feedback = self.summarizer.generate_feedback(prev_out["running"], self.trace)
            logger.log_object(feedback, tag="feedback")
            self.trace.hist.append((prev_out["running"], feedback))
