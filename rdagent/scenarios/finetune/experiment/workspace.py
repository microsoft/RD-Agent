"""
FT-specific Workspace implementation with minimal checkpoint strategy.

This module provides FTWorkspace, which configures checkpoint to only save
configuration files (train.yaml), excluding all training outputs.

Design Philosophy:
- Checkpoint is for code version control during CoSTEER evolution
- Model persistence is handled separately by Runner's save_model()
- This separation keeps concerns clear and checkpoints lightweight
"""

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.experiment import FBWorkspace


class FTWorkspace(FBWorkspace):
    """
    Fine-tuning workspace with minimal checkpoint strategy.

    Checkpoint Strategy:
    - Only saves configuration files (train.yaml) for version control
    - Training outputs (models, checkpoints) are excluded by design
    - Final model persistence is Runner's responsibility, not checkpoint's
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Configure checkpoint to only save configuration files
        # Training outputs are managed separately by save_final_model()
        RD_AGENT_SETTINGS.workspace_ckp_white_list_names = ["train.yaml"]
