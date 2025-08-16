import os
from pathlib import Path

import fire

from rdagent.app.finetune.llm.coder import LLMPipelineCoSTEER
from rdagent.app.finetune.llm.conf import FT_RD_SETTING, get_ft_env, update_settings
from rdagent.app.finetune.llm.prompts import (
    get_data_processing_prompt,
    get_finetuning_prompt,
    get_llamafactory_guide,
)
from rdagent.app.finetune.llm.tasks import create_llm_finetune_tasks
from rdagent.core.experiment import FBWorkspace
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.finetune.download import download_dataset, download_model
from rdagent.scenarios.finetune.utils import prev_model_dirname
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env


class SimpleLLMFinetuneLoop:
    """Simplified LLM fine-tuning loop with two steps: data processing -> fine-tuning"""

    def __init__(self, dataset: str, model: str):
        self.dataset = dataset
        self.model = model

        # Create scenario
        scen_class = import_class(FT_RD_SETTING.scen)
        self.scen = scen_class()

        # Create code generator
        self.coder = LLMPipelineCoSTEER(self.scen)

        # Create environment with volume mapping for data visibility
        data_volumes = {}
        if FT_RD_SETTING.local_data_path:
            # Input data should be read-only to protect original data
            data_volumes[FT_RD_SETTING.local_data_path] = {
                "bind": "/workspace/llm_finetune/data/raw",
                "mode": "ro",
            }

        # Ensure output directory is visible outside container and writable
        output_dir = Path.cwd() / "llm_finetune_output"
        output_dir.mkdir(exist_ok=True)
        data_volumes[str(output_dir)] = {
            "bind": "/workspace/llm_finetune/output",
            "mode": "rw",
        }

        self.env = get_ft_env(
            extra_volumes=data_volumes,
            running_timeout_period=None,  # No time limit
            enable_cache=False,
        )

        logger.info(f"Initialized simple LLM finetune loop for {model} on {dataset}")

    def run(self):
        """Run simplified fine-tuning pipeline"""
        logger.info("Starting simplified LLM fine-tuning pipeline...")

        # Step 1: Data processing
        logger.info("Step 1: Processing dataset...")
        data_exp = self._create_data_processing_experiment()
        data_exp = self.coder.develop(data_exp)
        self._execute_experiment(data_exp, "Data Processing")

        # Step 2: Model fine-tuning
        logger.info("Step 2: Fine-tuning model...")
        finetune_exp = self._create_finetuning_experiment()
        finetune_exp = self.coder.develop(finetune_exp)
        self._execute_experiment(finetune_exp, "Model Fine-tuning")

        logger.info("LLM fine-tuning pipeline completed!")

    def _create_data_processing_experiment(self) -> DSExperiment:
        """Create data processing experiment"""

        # Get runtime environment information
        runtime_info = get_runtime_environment_by_env(self.env)

        # Get dataset samples (simplified implementation)
        data_samples = self._get_dataset_samples()

        # Create data processing task
        task = create_llm_finetune_tasks(self.dataset, self.model)[0]  # First task is data processing

        # Set task description
        task.description = get_data_processing_prompt(self.dataset, runtime_info, data_samples)

        return DSExperiment(pending_tasks_list=[[task]])

    def _create_finetuning_experiment(self) -> DSExperiment:
        """Create fine-tuning experiment"""

        # Get runtime environment information
        runtime_info = get_runtime_environment_by_env(self.env)

        # Get LLaMA-Factory usage guide
        llamafactory_guide = get_llamafactory_guide()

        # Create fine-tuning task
        task = create_llm_finetune_tasks(self.dataset, self.model)[1]  # Second task is fine-tuning

        # Set task description
        task.description = get_finetuning_prompt(self.model, self.dataset, runtime_info, llamafactory_guide)

        return DSExperiment(pending_tasks_list=[[task]])

    def _execute_experiment(self, exp: DSExperiment, step_name: str):
        """Execute experiment"""
        logger.info(f"Executing {step_name}...")

        if not exp.is_ready_to_run():
            logger.error(f"{step_name} experiment is not ready to run")
            return

        # Execute experiment
        workspace = exp.experiment_workspace
        if workspace and hasattr(workspace, "run"):
            result = workspace.run(env=self.env, entry="python main.py")
            logger.info(f"{step_name} execution result: {result.exit_code}")
            if result.stdout:
                logger.info(f"{step_name} output:\n{result.stdout}")
        else:
            logger.warning(f"No executable workspace found for {step_name}")

    def _get_dataset_samples(self) -> str:
        """Get dataset samples (simplified implementation)"""
        try:
            # Try to read dataset samples from local path
            dataset_path = Path(FT_RD_SETTING.local_data_path) / self.dataset
            if dataset_path.exists():
                # Simple processing, return first few samples
                return f"Dataset path: {dataset_path}\nPlease load and analyze the dataset from this path."
            else:
                return f"Dataset {self.dataset} not found locally. Please download it first."
        except Exception as e:
            logger.warning(f"Could not load dataset samples: {e}")
            return f"Dataset: {self.dataset}\nPlease download and analyze the dataset."


def main(
    model: str | None = None,
    dataset: str | None = None,
    simple: bool = True,  # Default to simplified version
):
    """
    Simplified LLM fine-tuning entry point

    Parameters
    ----------
    dataset : str
        Dataset name for fine-tuning
    model : str
        Model name for fine-tuning
    simple : bool
        Whether to use simplified version (recommended)

    Example:
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh --model Qwen/Qwen2.5-1.5B-Instruct
    """
    if not dataset:
        raise Exception("Please specify dataset name.")

    ft_root_str = os.environ.get("FT_FILE_PATH")
    if not ft_root_str:
        raise Exception("Please set FT_FILE_PATH in your .env.")
    ft_root = Path(ft_root_str)
    if not ft_root.exists():
        raise Exception(f"FT_FILE_PATH does not exist: {ft_root}")

    # Ensure dataset and model exist
    ensure_ft_assets_exist(model, dataset, ft_root)
    update_settings(dataset, model)

    if simple:
        # Use simplified fine-tuning loop
        logger.info("Using simplified LLM fine-tuning pipeline")
        loop = SimpleLLMFinetuneLoop(dataset, model)
        loop.run()
    else:
        # Use original complex loop (backward compatibility)
        logger.info("Using original complex LLM fine-tuning pipeline")
        import asyncio

        from rdagent.scenarios.finetune.loop import FinetuneRDLoop

        rd_loop = FinetuneRDLoop(FT_RD_SETTING)
        asyncio.run(rd_loop.run())


def ensure_ft_assets_exist(model: str | None, dataset: str, ft_root: Path) -> None:
    """Ensure dataset and model assets exist under FT_FILE_PATH structure.

    - Dataset path: <ft_root>/dataset/<dataset>
    - Model path:   <ft_root>/model/<model>
    - Prev path:    <ft_root>/prev_model/<model>_<dataset>
    """
    dataset_dir = ft_root / "dataset" / dataset
    if not dataset_dir.exists():
        try:
            download_dataset(dataset, out_dir_root=str(ft_root / "dataset"))
        except Exception as e:
            raise Exception(f"Failed to download dataset '{dataset}' to {dataset_dir}: {e}") from e

    # Model may be optional for some flows, but for finetune we typically require one of prev_model or model
    if model is not None:
        prev_dir = ft_root / "prev_model" / prev_model_dirname(model, dataset)
        model_dir = ft_root / "model" / model
        if not prev_dir.exists() and not model_dir.exists():
            try:
                download_model(model, out_dir_root=str(ft_root / "model"))
            except Exception as e:
                raise Exception(
                    f"Failed to download model '{model}' to {model_dir}: {e}. "
                    f"At least one of prev_model or model is required."
                ) from e


if __name__ == "__main__":
    fire.Fire(main)
