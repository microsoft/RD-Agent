import asyncio
import json
from pathlib import Path
from typing import Any

from rdagent.app.finetune.llm.conf import LLMFinetunePropSetting
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import HypothesisFeedback, Trace
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.llama_factory_manager import LLaMAFactoryManager
from rdagent.scenarios.finetune.scen.utils import generate_dataset_info_config
from rdagent.utils.workflow import LoopBase


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
        self.trace = Trace(scen=scen)

        # Store finetune settings
        self.ft_rd_setting = PROP_SETTING
        self.dataset = PROP_SETTING.dataset
        self.model = PROP_SETTING.base_model

        # Setup environment
        self._setup_environment()

        # Initialize LLaMA Factory information manager
        self._get_llama_factory_info()

        # Generate dataset info during initialization
        # TODO: should we move this to the download step?
        self.get_dataset_info()

        # Initialize LoopBase (skip RDLoop.__init__ to avoid double initialization)
        LoopBase.__init__(self)

    def _setup_environment(self):
        """Setup Docker environment with standard finetune volume mappings"""
        self.env = get_ft_env(
            running_timeout_period=None,
            enable_cache=False,
        )

    def _get_llama_factory_info(self):
        """Setup LLaMA Factory information manager and extract information during initialization"""
        logger.info("Initializing LLaMA Factory information manager...")

        # Set cache directory to project root
        cache_dir = Path(self.ft_rd_setting.file_path) / ".llama_factory_info"
        self.llama_factory_manager = LLaMAFactoryManager(cache_dir)

        try:
            # Extract fresh information during initialization
            info = self.llama_factory_manager.get_info()

            # Verify if current model is supported
            if self.model not in self.llama_factory_manager.models:
                logger.warning(f"Model '{self.model}' is not in LLaMA Factory supported list")
                logger.info(
                    f"Supported models: {self.llama_factory_manager.models[:5]}... (total {len(self.llama_factory_manager.models)} models)"
                )

            # Record metadata information for debugging
            metadata_info = self.llama_factory_manager.get_metadata_info()
            if metadata_info.get("has_metadata"):
                commit_sha = metadata_info.get("commit_sha", "unknown")
                logger.info(f"Successfully extracted LLaMA Factory information (commit: {commit_sha})")
            else:
                logger.info("Successfully extracted LLaMA Factory information")

        except Exception as e:
            logger.error(f"LLaMA Factory information extraction failed: {e}")
            logger.warning("Will continue execution, but may affect some functionalities")
            # Do not throw exception, allow program to continue execution

    def get_dataset_info(self):
        """Generate dataset_info.json configuration for LLaMA-Factory compatibility"""
        # Path to the main dataset_info.json file
        datasets_dir = Path(self.ft_rd_setting.file_path) / "datasets"
        dataset_info_path = datasets_dir / "dataset_info.json"

        # Check if dataset configuration already exists
        existing_config = {}
        if dataset_info_path.exists():
            try:
                with open(dataset_info_path, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing dataset_info.json: {e}")
                existing_config = {}

        if self.dataset in existing_config:
            logger.info(f"Dataset '{self.dataset}' already configured in dataset_info.json, skipping")
            return

        logger.info(f"Generating dataset_info.json configuration for dataset '{self.dataset}'...")

        # Generate configuration using utility function
        generated_config = generate_dataset_info_config(self.dataset, self.ft_rd_setting.file_path)

        # Update the dataset_info.json file
        existing_config[self.dataset] = generated_config

        try:
            with open(dataset_info_path, "w", encoding="utf-8") as f:
                json.dump(existing_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully updated dataset_info.json with configuration for '{self.dataset}'")
        except Exception as e:
            raise RuntimeError(f"Failed to write dataset_info.json: {e}")

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        """Generate LLM fine-tuning experiment"""
        if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
            exp = self.hypothesis_gen.gen(self.trace)
            logger.log_object(exp.sub_tasks, tag="experiment generation")
            return exp

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
