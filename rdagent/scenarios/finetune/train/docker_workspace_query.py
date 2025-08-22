"""
Docker Workspace Query Module for LLM Finetune

This module provides functionality to query the actual Docker workspace structure
after data preprocessing but before training execution. This approach reuses the
FileTreeGenerator from data science scenario.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DockerWorkspaceQuery:
    """Query actual Docker workspace structure for LLM fine-tuning."""

    def __init__(self, shared_workspace_dir: Optional[Path] = None):
        """Initialize the Docker workspace query.

        Args:
            shared_workspace_dir: Path to the shared workspace directory
                                 (typically from LLMFinetuneRDLoop)
        """
        self.shared_workspace_dir = shared_workspace_dir

    def get_actual_workspace_structure(self) -> Optional[str]:
        """Get the actual workspace structure using FileTreeGenerator.

        This method can access the real file structure after data preprocessing
        but before Docker training execution.

        Returns:
            Formatted file tree string or None if not available
        """
        if not self.shared_workspace_dir or not self.shared_workspace_dir.exists():
            logger.warning(f"Shared workspace directory not available: {self.shared_workspace_dir}")
            return None

        try:
            from rdagent.scenarios.data_science.scen.utils import FileTreeGenerator

            # Create FileTreeGenerator with LLM finetune specific priority files
            priority_files = {".json", ".yaml", ".yml", ".csv", ".txt", ".md", ".py"}
            tree_gen = FileTreeGenerator(
                max_lines=100,  # Reasonable limit for prompt inclusion
                priority_files=priority_files,
                hide_base_name=True,
            )

            # Generate tree for shared workspace
            tree_structure = tree_gen.generate_tree(self.shared_workspace_dir)

            return tree_structure

        except Exception as e:
            logger.error(f"Failed to generate workspace tree: {e}")
            return None

    def get_docker_paths_mapping(self) -> Dict[str, str]:
        """Get mapping from local paths to Docker container paths.

        Returns:
            Dictionary mapping local paths to Docker paths
        """
        try:
            from rdagent.utils.env import LLMDockerConf

            docker_conf = LLMDockerConf()
            docker_mount = docker_conf.mount_path.rstrip("/")
        except Exception:
            docker_mount = "/workspace/llm_finetune"

        return {
            "local_shared": str(self.shared_workspace_dir) if self.shared_workspace_dir else "",
            "docker_shared": f"{docker_mount}/shared",
            "docker_data": f"{docker_mount}/data",
            "docker_output": f"{docker_mount}/output",
            "docker_working": docker_mount,
        }

    def format_workspace_info_for_prompt(self, dataset_name: Optional[str] = None) -> str:
        """Format workspace information for LLM prompt.

        Args:
            dataset_name: Optional dataset name for context

        Returns:
            Formatted workspace information string
        """
        lines = ["## Docker Container Environment Information"]

        # Get Docker paths
        paths = self.get_docker_paths_mapping()
        if paths["docker_working"]:
            lines.extend(
                [
                    f"**Working Directory**: `{paths['docker_working']}/`",
                    f"**Data Directory**: `{paths['docker_data']}/`",
                    f"**Output Directory**: `{paths['docker_output']}/`",
                    f"**Shared Directory**: `{paths['docker_shared']}/`",
                    "",
                ]
            )

        # Try to get actual file structure
        actual_structure = self.get_actual_workspace_structure()
        if actual_structure:
            lines.extend(
                [
                    "**Actual Preprocessed Workspace Structure**:",
                    "```",
                    actual_structure,
                    "```",
                    "",
                    "**Key Files Available**:",
                    "- `processed_dataset.json`: Preprocessed training data in LlamaFactory format",
                    "- `dataset_info.json`: Dataset configuration mapping for LlamaFactory",
                    "",
                ]
            )
        else:
            # Fallback to expected structure
            lines.extend(
                [
                    "**Expected Docker Container Structure**:",
                    "```",
                    f"{paths['docker_working']}/",
                    "├── data/",
                    "│   ├── dataset_info.json        # LlamaFactory dataset configuration",
                    "│   └── processed_dataset.json   # Preprocessed training data",
                    "├── shared/",
                    "│   ├── dataset_info.json        # Dataset configuration (copy)",
                    "│   └── processed_dataset.json   # Preprocessed data (copy)",
                    "└── output/                      # Training output directory",
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                "**Critical Configuration Requirements**:",
                f'- dataset: `"processed_dataset"` (string, not dictionary)',
                f"- dataset_dir: `\"{paths['docker_data']}\"`",
                f"- output_dir: `\"{paths['docker_output']}\"`",
            ]
        )

        return "\n".join(lines)

    def validate_preprocessed_files(self) -> List[str]:
        """Validate that required preprocessed files exist.

        Returns:
            List of missing files (empty if all files exist)
        """
        if not self.shared_workspace_dir:
            return ["shared_workspace_dir not available"]

        missing_files = []
        required_files = ["processed_dataset.json", "dataset_info.json"]

        for file_name in required_files:
            file_path = self.shared_workspace_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)

        return missing_files


def get_workspace_info_for_prompt(
    shared_workspace_dir: Optional[Path] = None, dataset_name: Optional[str] = None
) -> str:
    """Get workspace information formatted for LLM prompt.

    Args:
        shared_workspace_dir: Path to shared workspace directory
        dataset_name: Optional dataset name

    Returns:
        Formatted workspace information
    """
    query = DockerWorkspaceQuery(shared_workspace_dir)
    return query.format_workspace_info_for_prompt(dataset_name)


# Example usage
if __name__ == "__main__":
    # Test with a mock path
    test_path = Path("/tmp/test_workspace")
    test_path.mkdir(exist_ok=True)

    # Create some test files
    (test_path / "processed_dataset.json").write_text('{"test": "data"}')
    (test_path / "dataset_info.json").write_text('{"processed_dataset": {"file_name": "processed_dataset.json"}}')

    query = DockerWorkspaceQuery(test_path)

    print("=== Workspace Structure Query Test ===")
    structure = query.get_actual_workspace_structure()
    if structure:
        print("Actual structure:")
        print(structure)
    else:
        print("Could not get actual structure")

    print("\n=== Prompt Format Test ===")
    prompt_info = query.format_workspace_info_for_prompt("test-dataset")
    print(prompt_info)

    print("\n=== File Validation Test ===")
    missing = query.validate_preprocessed_files()
    print(f"Missing files: {missing if missing else 'None - all files present!'}")

    # Cleanup
    import shutil

    shutil.rmtree(test_path, ignore_errors=True)
