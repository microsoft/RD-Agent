"""
Streamlined LLaMA Factory manager for parameter extraction.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import requests

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger

EXTRACT_PARAMETERS_SCRIPT_NAME = "extract_parameters.py"

# Regex patterns to exclude parameters not relevant for SFT training prompts
EXCLUDED_PARAM_PATTERNS = [
    # Inference engines
    r"^infer_",  # Inference related
    r"^vllm_",  # vLLM engine
    r"^sglang_",  # SGLang engine
    r"^kt_",  # KTransformers config (kt_maxlen, kt_mode, etc.)
    r"^use_kt$",  # KTransformers toggle
    r"^use_kv_cache$",  # Inference only
    r"^cpu_infer$",  # KTransformers: CPU cores for computation
    r"^chunk_size$",  # KTransformers: chunk size for CPU compute
    # Hub/Cloud
    r"^push_to_hub",  # Hub push
    r"^hub_",  # Hub related
    r"_hub_token$",  # Hub tokens (hf_hub_token, ms_hub_token, om_hub_token)
    # Multimodal inputs (text-only SFT)
    r"^image_",  # Image inputs
    r"^video_",  # Video inputs
    r"^audio_",  # Audio inputs
    r"^crop_to_patches$",  # Image processing for internvl
    r"^use_audio_in_video$",  # Video audio
    r"^media_dir$",  # Media directory for multimodal
    # Export (post-training)
    r"^export_",  # Model export
    # Hardware specific
    r"^tpu_",  # TPU related (tpu_num_cores, tpu_metrics_debug)
    # Third-party logging
    r"^ray_",  # Ray hyperparameter search
    r"^swanlab_",  # SwanLab logging
    r"^use_swanlab$",  # SwanLab toggle
    r"^trackio_",  # Trackio logging
    # RLHF/DPO (not for SFT)
    r"^pref_",  # Preference learning (DPO/KTO/ORPO/SimPO)
    r"^dpo_",  # DPO specific
    r"^kto_",  # KTO specific
    r"^simpo_",  # SimPO specific
    r"^ppo_",  # PPO specific
    r"^ref_model",  # Reference model for RLHF
    r"^reward_model",  # Reward model for PPO
    r"^ld_alpha$",  # LD-DPO
    # Deprecated (per help text)
    r"^no_cuda$",  # Deprecated in transformers 5.0
    r"^use_mps_device$",  # Deprecated in transformers 5.0
    r"^per_gpu_",  # Deprecated: use per_device_* instead
    # Unsloth (third-party, not used by default)
    r"^use_unsloth",  # use_unsloth, use_unsloth_gc
]
EXCLUDED_PARAM_REGEX = re.compile("|".join(EXCLUDED_PARAM_PATTERNS))


class LLaMAFactoryManager:
    """Manager for LLaMA Factory parameter extraction and caching."""

    def __init__(self):
        """Initialize the manager instance."""
        self.cache_dir = Path(FT_RD_SETTING.file_path) / ".llama_factory_info"
        self._info_cache: Optional[Dict] = None

    def extract_info_from_docker(self) -> Dict:
        """Extract LLaMA Factory information from Docker environment."""
        if not self.cache_dir.exists() or not any(self.cache_dir.iterdir()):
            logger.info("Extract LLaMA Factory parameters from Docker")
            # Prepare extraction script
            workspace = FBWorkspace()
            script_path = Path(__file__).parent / "docker_scripts" / EXTRACT_PARAMETERS_SCRIPT_NAME
            workspace.inject_files(**{EXTRACT_PARAMETERS_SCRIPT_NAME: script_path.read_text()})

            # Setup cache directory and Docker volumes
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            volumes = {str(self.cache_dir): {"bind": "/workspace/.llama_factory_info", "mode": "rw"}}

            # Run extraction
            result = workspace.run(
                env=get_ft_env(extra_volumes=volumes, running_timeout_period=120, enable_cache=False),
                entry=f"python {EXTRACT_PARAMETERS_SCRIPT_NAME}",
            )

            if result.exit_code != 0:
                raise RuntimeError(f"Parameter extraction failed: {result.stdout}")

        else:
            logger.info("Skip updating LLaMA Factory, using local cache")

        # Load the extracted data
        self._info_cache = self._load_extracted_data()
        if not self._info_cache:
            raise RuntimeError("Failed to load extracted LLaMA Factory information")

        logger.info("Successfully extracted LLaMA Factory parameters")
        return self._info_cache

    def _load_extracted_data(self) -> Dict:
        """Load extracted information from flat file structure."""
        data = {}

        # Load constants
        constants_file = self.cache_dir / "constants.json"
        if constants_file.exists():
            with open(constants_file, encoding="utf-8") as f:
                data.update(json.load(f))

        # Load parameters
        parameters_file = self.cache_dir / "parameters.json"
        if parameters_file.exists():
            with open(parameters_file, encoding="utf-8") as f:
                data["parameters"] = json.load(f)

        return data

    def get_info(self) -> Dict:
        """Get complete LLaMA Factory information, extracting on first call."""
        if self._info_cache is None:
            self._info_cache = self.extract_info_from_docker()
        return self._info_cache

    @property
    def methods(self) -> List[str]:
        """Available fine-tuning methods."""
        return self.get_info().get("methods", [])

    @property
    def models(self) -> List[str]:
        """Available base models."""
        return list(self.get_info().get("supported_models", {}).keys())

    @property
    def hf_models(self) -> List[str]:
        """Available HuggingFace models."""
        supported_models = self.get_info().get("supported_models", {})
        hf_model_set = set()
        for hf_model in supported_models.values():
            if isinstance(hf_model, str):
                hf_model_set.add(hf_model)
        return list(hf_model_set)

    @property
    def peft_methods(self) -> List[str]:
        """Available PEFT methods, dynamically filtered from available methods."""
        known_peft = {"lora", "qlora", "adalora"}
        return [m for m in self.methods if m in known_peft]

    @property
    def training_stages(self) -> Dict[str, str]:
        """Training stage mapping."""
        return self.get_info().get("training_stages", {})

    @property
    def templates(self) -> List[str]:
        """Available chat templates."""
        return self.get_info().get("templates", [])

    def get_template_for_model(self, model_name: str) -> Optional[str]:
        """Get template for model. Returns None to let LlamaFactory auto-detect.

        Args:
            model_name: Model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct")

        Returns:
            None - LlamaFactory will automatically detect the appropriate template
        """
        return None

    def is_peft_method(self, method: str) -> bool:
        """Check if the given method is a PEFT method."""
        return method in self.peft_methods

    def get_parameters(self, param_type: Optional[str] = None) -> Dict:
        """Get parameters by type or all parameters."""
        params = self.get_info().get("parameters", {})
        if param_type:
            return params.get(param_type, {})
        return params

    def _format_param_line(self, param_name: str, param_info: dict, truncate_help: bool = True) -> str:
        """Format a single parameter line (extracted common logic)."""
        help_text = param_info["help"][:80] if truncate_help else param_info["help"]
        type_text = param_info.get("type", "").replace("typing.", "")
        default_val = param_info.get("default")

        param_line = f"- {param_name}"
        if type_text or default_val is not None:
            param_line += " ("
            if type_text:
                param_line += f"{type_text}"
            if default_val is not None:
                param_line += f", default={default_val}" if type_text else f"default={default_val}"
            param_line += ")"
        return param_line + f": {help_text}"

    def _format_params_dict(self, params_dict: dict, truncate_help: bool = True) -> list[str]:
        """Format a dictionary of parameters (extracted common logic)."""
        return [
            self._format_param_line(name, info, truncate_help)
            for name, info in params_dict.items()
            if isinstance(info, dict) and "help" in info and not EXCLUDED_PARAM_REGEX.search(name)
        ]

    def format_shared_params(self, truncate_help: bool = True) -> str:
        """Format shared parameters (model, data, training) that apply to all methods."""
        all_params = self.get_parameters()
        sections = []

        for param_type in ["model", "data", "training"]:
            if param_type in all_params:
                sections.append(f"### {param_type.upper()} Parameters:")
                sections.extend(self._format_params_dict(all_params[param_type], truncate_help))
                sections.append("")  # Empty line

        return "\n".join(sections).rstrip()

    def format_method_specific_params(self, method: str, truncate_help: bool = True) -> str:
        """Format only method-specific finetuning parameters."""
        all_params = self.get_parameters()
        if "finetuning" not in all_params:
            return f"**{method}**: No specific parameters"

        finetuning_params = all_params["finetuning"]
        method_lower = method.lower()

        # Full fine-tuning has no PEFT-specific parameters
        if method_lower == "full":
            return f"**{method}**: Uses shared parameters only (full-parameter training)"

        # Get parameters directly from the structured finetuning_params by method name
        if method_lower in finetuning_params:
            type_params = finetuning_params[method_lower]
        else:
            # Unknown method, return message
            return f"**{method}**: No specific parameters found for this method"

        if not type_params:
            return f"**{method}**: Uses shared parameters only"

        lines = [f"**{method}**:"]
        lines.extend(self._format_params_dict(type_params, truncate_help))
        return "\n".join(lines)


LLaMAFactory_manager = LLaMAFactoryManager()
