import asyncio
from pathlib import Path

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


def main(
    model: str | None = None,
    dataset: str | None = None,
):
    """
    Parameters
    ----------
    dataset :
        Dateset name, used for finetune.

    Auto R&D Evolving loop for models finetune.
    You can continue running a session by using the command:
    .. code-block:: bash
        dotenv run -- python rdagent/app/finetune/loop.py --dataset shibing624/alpaca-zh
    """
    if not dataset:
        raise Exception("Please specify dataset name.")

    model_folder = Path(DS_RD_SETTING.local_data_path) / dataset / "prev_model"
    if not model_folder.exists():
        # NOTE: must be superuser to use mount.
        raise Exception(f"Please put the model path to {model_folder}.")

    # NOTE: due to finetune application
    DS_RD_SETTING.scen = "rdagent.app.finetune.scen.LLMFinetuneScen"
    DS_RD_SETTING.hypothesis_gen = "rdagent.app.finetune.proposal.FinetuneExpGen"
    DS_RD_SETTING.competition = dataset
    DS_RD_SETTING.enable_model_dump = True
    DS_RD_SETTING.debug_timeout = 3600 * 10
    DS_RD_SETTING.full_timeout = 3600 * 100
    rd_loop: DataScienceRDLoop = DataScienceRDLoop(DS_RD_SETTING)
    asyncio.run(rd_loop.run())


if __name__ == "__main__":
    fire.Fire(main)
