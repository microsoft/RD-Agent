"""
FT-specific Workspace implementation with optimized checkpoint handling.

This module provides FTWorkspace, which extends FBWorkspace with special handling
for LLM fine-tuning scenarios where training outputs (large model files) should be
excluded from checkpoints.
"""

import shutil
from pathlib import Path

from rdagent.components.coder.finetune.conf import get_clear_ws_cmd
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger


class FTWorkspace(FBWorkspace):
    """
    Fine-tuning specialized workspace that cleans large training outputs before checkpointing.

    Problem: LLM training generates huge files (models, optimizer states) that shouldn't be
    in checkpoints. These files (up to 24GB+) cause checkpoint creation to hang.

    Solution: Override create_ws_ckp to clean training outputs before creating checkpoint.
    This ensures checkpoint only contains code and small config files.
    """

    def create_ws_ckp(self) -> None:
        """
        Create workspace checkpoint after cleaning large training outputs.

        This method:
        1. Removes large training artifacts (model weights, checkpoints, optimizer states)
        2. Calls parent's create_ws_ckp to create a lightweight checkpoint
        3. Restores the cleaned files from file_dict if needed

        Files removed before checkpoint:
        - output/ directory (contains full trained models)
        - checkpoint-* directories (intermediate checkpoints with optimizer states)
        - *.safetensors, *.bin (model weight files)
        - adapter_*, training_*.json, *_metrics.json (training artifacts)
        """
        # Record what we're cleaning
        cleaned_items = []

        # Pattern matching for files/dirs to clean
        # These match the patterns in get_clear_ws_cmd("before_training")
        cleanup_patterns = [
            "output",
            "checkpoint-*",
            "adapter_*",
            "*.safetensors",
            "*.bin",
            "training_*.json",
            "*_metrics.json",
            "micro_test_output",  # Also clean debug output
        ]

        # Clean large files before checkpoint
        for pattern in cleanup_patterns:
            matched_paths = list(self.workspace_path.glob(pattern))
            for path in matched_paths:
                try:
                    if path.is_dir():
                        size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024
                        shutil.rmtree(path)
                        cleaned_items.append(f"{path.name}/ ({size_mb:.1f}MB)")
                    elif path.is_file():
                        size_mb = path.stat().st_size / 1024 / 1024
                        path.unlink()
                        cleaned_items.append(f"{path.name} ({size_mb:.1f}MB)")
                except Exception as e:
                    logger.warning(f"Failed to clean {path}: {e}")

        if cleaned_items:
            logger.info(
                f"FTWorkspace: Cleaned {len(cleaned_items)} training artifacts before checkpoint: "
                f"{', '.join(cleaned_items[:3])}"
                + (f" and {len(cleaned_items)-3} more" if len(cleaned_items) > 3 else "")
            )

        # Now create checkpoint with clean workspace
        super().create_ws_ckp()

        # Note: We don't restore the cleaned files because:
        # 1. Training outputs are recreated on each run
        # 2. Keeping them would defeat the purpose of cleaning
        # 3. file_dict only contains injected code, not training outputs
