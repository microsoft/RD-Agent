"""
Docker Environment Query Module for LLM Finetune

This module provides functionality to query Docker environment structure for LLM fine-tuning tasks.
Since the coder runs outside Docker but needs to know the Docker environment, this module
provides the expected Docker file structure without actually running in Docker.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DockerEnvStructureQuery:
    """Query Docker environment structure for LLM fine-tuning."""

    def __init__(self):
        """Initialize the Docker environment structure query."""
        self.docker_base_path = "/workspace/llm_finetune"

    def get_expected_docker_structure(self, dataset_name: Optional[str] = None) -> Dict[str, str]:
        """Get the expected Docker container file structure.

        Args:
            dataset_name: Optional dataset name for dynamic path generation

        Returns:
            Dictionary mapping logical names to Docker paths
        """
        base = self.docker_base_path
        structure = {
            "working_dir": f"{base}/",
            "data_dir": f"{base}/data/",
            "output_dir": f"{base}/output/",
            "shared_dir": f"{base}/shared/",
            "processed_dataset": f"{base}/data/processed_dataset.json",
            "dataset_info": f"{base}/data/dataset_info.json",
            "shared_processed_dataset": f"{base}/shared/processed_dataset.json",
            "shared_dataset_info": f"{base}/shared/dataset_info.json",
        }

        if dataset_name:
            structure["raw_dataset_dir"] = f"{base}/data/raw/{dataset_name}/"

        return structure

    def get_docker_mount_info(self) -> Dict[str, str]:
        """Get Docker mount information from configuration.

        Returns:
            Dictionary with mount path information
        """
        try:
            from rdagent.utils.env import LLMDockerConf

            docker_conf = LLMDockerConf()
            return {
                "mount_path": docker_conf.mount_path,
                "image": docker_conf.image,
                "default_entry": docker_conf.default_entry,
            }
        except Exception as e:
            logger.warning(f"Could not get Docker configuration: {e}")
            return {
                "mount_path": "/workspace/llm_finetune/",
                "image": "local_llm_finetune:latest",
                "default_entry": "llamafactory-cli version",
            }

    def format_docker_paths_for_prompt(self, dataset_name: Optional[str] = None) -> str:
        """Format Docker paths for inclusion in LLM prompt.

        Args:
            dataset_name: Optional dataset name

        Returns:
            Formatted string with Docker path information
        """
        structure = self.get_expected_docker_structure(dataset_name)
        mount_info = self.get_docker_mount_info()

        lines = [
            "## Docker Container Environment",
            f"**Container Image**: {mount_info['image']}",
            f"**Mount Path**: {mount_info['mount_path']}",
            "",
            "**Expected File Structure in Docker Container**:",
            "```",
            f"{structure['working_dir']}                    # Main working directory",
            f"├── data/                                   # Dataset directory",
            f"│   ├── dataset_info.json                   # LlamaFactory dataset configuration",
            f"│   ├── processed_dataset.json              # Preprocessed training data",
        ]

        if dataset_name:
            lines.append(f"│   └── raw/{dataset_name}/             # Raw dataset files")
        else:
            lines.append(f"│   └── raw/<dataset>/                  # Raw dataset files")

        lines.extend(
            [
                f"├── shared/                                 # Shared data processing outputs",
                f"│   ├── dataset_info.json                   # Dataset configuration (copy)",
                f"│   └── processed_dataset.json              # Preprocessed data (copy)",
                f"├── output/                                 # Training output directory",
                f"└── workspace_cache/                        # Workspace cache (if enabled)",
                "```",
                "",
                "**Critical Docker Paths** (use these exact paths in your configuration):",
                f"- Working Directory: `{structure['working_dir']}`",
                f"- Dataset Directory: `{structure['data_dir']}`",
                f"- Processed Dataset: `{structure['processed_dataset']}`",
                f"- Dataset Configuration: `{structure['dataset_info']}`",
                f"- Output Directory: `{structure['output_dir']}`",
            ]
        )

        return "\n".join(lines)

    def get_critical_config_rules(self) -> List[str]:
        """Get critical configuration rules for Docker environment.

        Returns:
            List of critical configuration rules
        """
        structure = self.get_expected_docker_structure()

        return [
            f'dataset: MUST be "processed_dataset" (string, not dictionary)',
            f'dataset_dir: MUST be "{structure["data_dir"].rstrip("/")}"',
            f'output_dir: MUST be "{structure["output_dir"].rstrip("/")}"',
            "model_name_or_path: Use HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')",
            "All file paths must use Docker container paths, NOT local filesystem paths",
        ]

    def validate_config_paths(self, config_dict: Dict) -> List[str]:
        """Validate configuration paths against Docker environment.

        Args:
            config_dict: Configuration dictionary to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        structure = self.get_expected_docker_structure()

        # Check dataset configuration
        if config_dict.get("dataset") != "processed_dataset":
            errors.append(f"dataset should be 'processed_dataset', got: {config_dict.get('dataset')}")

        # Check dataset_dir
        expected_data_dir = structure["data_dir"].rstrip("/")
        if config_dict.get("dataset_dir") != expected_data_dir:
            errors.append(f"dataset_dir should be '{expected_data_dir}', got: {config_dict.get('dataset_dir')}")

        # Check output_dir
        expected_output_dir = structure["output_dir"].rstrip("/")
        if config_dict.get("output_dir") != expected_output_dir:
            errors.append(f"output_dir should be '{expected_output_dir}', got: {config_dict.get('output_dir')}")

        return errors


# Convenience function
def get_docker_env_info_for_prompt(dataset_name: Optional[str] = None) -> str:
    """Get Docker environment information formatted for LLM prompt.

    Args:
        dataset_name: Optional dataset name

    Returns:
        Formatted Docker environment information
    """
    query = DockerEnvStructureQuery()
    return query.format_docker_paths_for_prompt(dataset_name)


def get_critical_docker_rules() -> List[str]:
    """Get critical Docker configuration rules.

    Returns:
        List of critical configuration rules
    """
    query = DockerEnvStructureQuery()
    return query.get_critical_config_rules()


# Example usage
if __name__ == "__main__":
    # Test the functionality
    query = DockerEnvStructureQuery()

    print("=== Docker Environment Structure ===")
    print(query.format_docker_paths_for_prompt("alpaca-zh"))

    print("\n=== Critical Rules ===")
    for rule in query.get_critical_config_rules():
        print(f"- {rule}")

    print("\n=== Validation Test ===")
    test_config = {
        "dataset": "processed_dataset",
        "dataset_dir": "/workspace/llm_finetune/data",
        "output_dir": "/workspace/llm_finetune/output",
    }
    errors = query.validate_config_paths(test_config)
    print(f"Validation errors: {errors if errors else 'None - Config is valid!'}")
