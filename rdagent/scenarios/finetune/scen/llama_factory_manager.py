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
DEFAULT_HELP_TRUNCATE_LEN = None  # Default max length for help text in formatted output

# Regex patterns to exclude parameters not relevant for SFT training prompts
EXCLUDED_PARAM_PATTERNS = [
    # Inference engines & inference-only params
    r"^infer_",  # Inference related (infer_backend, infer_dtype)
    r"^vllm_",  # vLLM engine
    r"^sglang_",  # SGLang engine
    r"^kt_",  # KTransformers config (kt_maxlen, kt_mode, etc.)
    r"^use_kt$",  # KTransformers toggle
    r"^use_kv_cache$",  # Inference only
    r"^use_cache$",  # KV cache for generation
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
    r"^freeze_vision_tower$",  # MLLM: freeze vision encoder
    r"^freeze_multi_modal_projector$",  # MLLM: freeze projector
    r"^freeze_language_model$",  # MLLM: freeze LLM backbone
    # Export (post-training)
    r"^export_",  # Model export
    # Hardware specific (non-NVIDIA)
    r"^tpu_",  # TPU related (tpu_num_cores, tpu_metrics_debug)
    r"^use_cpu$",  # CPU-only training
    r"^use_ipex$",  # Intel Extension for PyTorch
    r"^jit_mode_eval$",  # PyTorch JIT for inference
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
    r"^torchdynamo$",  # Deprecated: use torch_compile_backend
    r"^fp16_backend$",  # Deprecated: use half_precision_backend
    r"^include_inputs_for_metrics$",  # Deprecated: use include_for_metrics
    # Unsloth (third-party, not used by default)
    r"^use_unsloth",  # use_unsloth, use_unsloth_gc
    # Internal/derived params (help says "Do not specify it")
    r"^compute_dtype$",
    r"^device_map$",
    r"^model_max_length$",
    r"^block_diag_attn$",
    # Platform-specific / internal
    r"^mp_parameters$",  # SageMaker launcher only
    r"^_n_gpu$",  # Internal variable
    r"^use_legacy_prediction_loop$",  # Legacy feature
    r"^past_index$",  # Rarely used
    r"^print_param_status$",  # Debug only
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
            env = get_ft_env(extra_volumes=volumes, enable_cache=False)
            env.conf.running_timeout_period = 120  # Short timeout for parameter extraction
            result = workspace.run(
                env=env,
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
        return list({v for v in supported_models.values() if isinstance(v, str)})

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

    def is_peft_method(self, method: str) -> bool:
        """Check if the given method is a PEFT method."""
        return method in self.peft_methods

    def get_parameters(self, param_type: Optional[str] = None) -> Dict:
        """Get parameters by type or all parameters."""
        params = self.get_info().get("parameters", {})
        if param_type:
            return params.get(param_type, {})
        return params

    def _format_param_line(self, param_name: str, param_info: dict, max_help_len: int | None) -> str:
        """Format a single parameter line.

        Args:
            max_help_len: Max length for help text. None means no truncation.
        """
        help_text = param_info["help"]
        if max_help_len:
            help_text = help_text[:max_help_len]
        type_text = param_info.get("type", "").replace("typing.", "")
        default_val = param_info.get("default")

        # Build metadata: filter out empty parts, join with comma
        parts = [p for p in [type_text, f"default={default_val}" if default_val is not None else ""] if p]
        meta = f" ({', '.join(parts)})" if parts else ""
        return f"- {param_name}{meta}: {help_text}"

    def _format_params_dict(self, params_dict: dict, max_help_len: int | None) -> list[str]:
        """Format a dictionary of parameters."""
        return [
            self._format_param_line(name, info, max_help_len)
            for name, info in params_dict.items()
            if isinstance(info, dict) and "help" in info and not EXCLUDED_PARAM_REGEX.search(name)
        ]

    def format_shared_params(self, max_help_len: int | None = DEFAULT_HELP_TRUNCATE_LEN) -> str:
        """Format shared parameters (model, data, training) that apply to all methods.

        Args:
            max_help_len: Max length for help text. None means no truncation.
        """
        all_params = self.get_parameters()
        sections = []

        for param_type in ["model", "data", "training"]:
            if param_type in all_params:
                sections.append(f"### {param_type.upper()} Parameters:")
                sections.extend(self._format_params_dict(all_params[param_type], max_help_len))
                sections.append("")

        return "\n".join(sections).rstrip()

    def format_method_specific_params(self, method: str, max_help_len: int | None = DEFAULT_HELP_TRUNCATE_LEN) -> str:
        """Format only method-specific finetuning parameters.

        Args:
            max_help_len: Max length for help text. None means no truncation.
        """
        all_params = self.get_parameters()
        if "finetuning" not in all_params:
            return f"**{method}**: No specific parameters"

        finetuning_params = all_params["finetuning"]
        method_lower = method.lower()

        if method_lower == "full":
            return f"**{method}**: Uses shared parameters only (full-parameter training)"

        if method_lower not in finetuning_params or not finetuning_params[method_lower]:
            return f"**{method}**: Uses shared parameters only"

        lines = [f"**{method}**:"]
        lines.extend(self._format_params_dict(finetuning_params[method_lower], max_help_len))
        return "\n".join(lines)


LLaMAFactory_manager = LLaMAFactoryManager()
