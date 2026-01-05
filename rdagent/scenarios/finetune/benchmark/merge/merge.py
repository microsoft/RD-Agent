import json
import subprocess
from pathlib import Path

from rdagent.components.coder.finetune.conf import get_workspace_prefix
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T

BLACKWELL_GPU_KEYWORDS = ["b100", "b200", "b300"]


def is_blackwell_gpu() -> bool:
    """Check if the current GPU is NVIDIA Blackwell architecture (B100, B200, B300)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpu_names = result.stdout.strip().lower()
            return any(kw in gpu_names for kw in BLACKWELL_GPU_KEYWORDS)
    except Exception:
        pass
    return False


def check_if_merging_needed(model_path: str | Path) -> bool:
    """
    Check if the model needs to be merged before benchmarking.
    Usually required when LoRA adapter has modules_to_save which vLLM doesn't support.
    """
    config_path = Path(model_path) / "adapter_config.json"
    if not config_path.exists():
        return False
    with open(config_path, "r") as f:
        config = json.load(f)
    # Check for modules_to_save which requires merging for vLLM
    # The logic is based in https://github.com/vllm-project/vllm/issues/9280
    if config.get("modules_to_save") is not None:
        logger.info(f"Model merging required due to modules_to_save: {config.get('modules_to_save')}")
        return True
    if is_blackwell_gpu():
        logger.info("Model merging required due to Blackwell GPU (B100/B200/B300)")
        return True
    return False


def merge_model(env, workspace_path: Path, base_model_path: str, adapter_path: str, output_path: str):
    """
    Merge LoRA adapter into base model using a template-generated script.
    """
    # Prepare template variables
    template_vars = {
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "output_path": output_path,
    }

    # Render Jinja2 template
    merge_script = T("rdagent.scenarios.finetune.benchmark.merge.merge_model_template:template").r(**template_vars)

    script_path = workspace_path / "merge_model.py"
    script_path.write_text(merge_script)

    logger.info(f"Starting model merging from {adapter_path}...")

    ws_prefix = get_workspace_prefix(env)
    cmd = f"python {ws_prefix}/merge_model.py"

    result = env.run(cmd, local_path=str(workspace_path))
    if result.exit_code != 0:
        raise RuntimeError(f"Model merging failed (exit_code={result.exit_code}):\n{result.stdout}")
    logger.info("Model merging completed.")
