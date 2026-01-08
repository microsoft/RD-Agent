import asyncio
from pathlib import Path

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.app.finetune.data_science.conf import update_settings
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


def main(
    model: str | None = None,
    competition: str | None = None,
):
    """
    Parameters
    ----------
    competition :
        Competition name.

    Auto R&D Evolving loop for models finetune.
    You can continue running a session by using the command:
    .. code-block:: bash
        dotenv run -- python rdagent/app/finetune/data_science/loop.py --competition aerial-cactus-identification
    """
    if not competition:
        raise Exception("Please specify competition name.")

    model_folder = Path(DS_RD_SETTING.local_data_path) / competition / "prev_model"
    if not model_folder.exists():
        raise Exception(f"Please put the model path to {model_folder}.")
    update_settings(competition)
    rd_loop: DataScienceRDLoop = DataScienceRDLoop(DS_RD_SETTING)
    asyncio.run(rd_loop.run())


if __name__ == "__main__":
    fire.Fire(main)
