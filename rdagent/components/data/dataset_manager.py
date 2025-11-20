"""Dataset migration and storage management."""

import shutil
from pathlib import Path
from typing import Any

from rdagent.log import rdagent_logger as logger


class DatasetManager:
    """Manage dataset migration and storage organization."""

    def __init__(self, permanent_root: str = "./datasets"):
        """
        Initialize dataset manager with storage paths.

        Args:
            permanent_root: Root directory for permanent storage (default: ./datasets)
        """
        self.permanent_root = Path(permanent_root).expanduser().resolve()
        self.raw_dir = self.permanent_root / "raw"
        self.converted_dir = self.permanent_root / "converted"

        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.converted_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")

    def migrate_dataset_selective(
        self,
        source_path: str,
        dataset_id: str,
        file_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Selectively migrate dataset (only copy useful files).

        Args:
            source_path: Source path (e.g., /tmp/dataset_staging/xxx)
            dataset_id: Dataset ID (e.g., Qwen/QA-math)
            file_analysis: File analysis result from DatasetInspector.analyze_files_for_sft()

        Returns:
            {
                "target_path": str,
                "copied_files": List[str],
                "skipped_files": List[str],
                "size_before_mb": float,
                "size_after_mb": float,
                "space_saved_mb": float
            }

        Raises:
            ValueError: If source path doesn't exist
        """
        source = Path(source_path)

        if not source.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

        # Target path
        target = self.raw_dir / dataset_id

        # If target exists, remove it first
        if target.exists():
            logger.warning(f"Target path already exists, will overwrite: {target}")
            shutil.rmtree(target)

        # Create target directory
        target.mkdir(parents=True, exist_ok=True)

        # Copy only useful files
        useful_files = file_analysis["useful_files"]
        copied_files = []
        skipped_files = file_analysis["junk_files"]

        logger.info(f"Migrating {len(useful_files)} useful files, skipping {len(skipped_files)} junk files")

        for file_rel_path in useful_files:
            source_file = source / file_rel_path
            target_file = target / file_rel_path

            if not source_file.exists():
                logger.warning(f"Source file not found (skipping): {source_file}")
                continue

            # Create parent directories if needed
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if source_file.is_file():
                shutil.copy2(source_file, target_file)
                copied_files.append(file_rel_path)
            elif source_file.is_dir():
                # Copy directory recursively
                shutil.copytree(source_file, target_file, dirs_exist_ok=True)
                copied_files.append(file_rel_path)

        logger.info(f"✅ Migration complete: {source} → {target}")
        logger.info(f"Copied {len(copied_files)} files, saved {file_analysis['space_saved_mb']:.2f}MB")

        return {
            "target_path": str(target),
            "copied_files": copied_files,
            "skipped_files": skipped_files,
            "size_before_mb": file_analysis["total_size_mb"],
            "size_after_mb": file_analysis["size_after_cleanup_mb"],
            "space_saved_mb": file_analysis["space_saved_mb"],
        }
