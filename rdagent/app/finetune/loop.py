import asyncio
from pathlib import Path

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


def main(
    base_model: str | None = None,
    dataset: str | None = None,
):
    """

    Parameters
    ----------
    base_model :
        A path like `./git_ignore_folder/qwen3`. This is the base model path.
    dataset :
        Dateset name, used for finetune.

    Auto R&D Evolving loop for models finetune.
    You can continue running a session by using the command:
    .. code-block:: bash
        dotenv run -- python rdagent/app/finetune/loop.py --base_model git_ignore_folder/qwen3 --dataset shibing624/alpaca-zh.
    """
    if not base_model or dataset:
        logger.error("Please specify base model path and dataset name.")

    # NOTE: due to finetune application
    DS_RD_SETTING.scen = "rdagent.app.finetune.scen.LLMFinetuneScen"
    DS_RD_SETTING.hypothesis_gen = "rdagent.app.finetune.proposal.FinetuneExpGen"
    DS_RD_SETTING.competition = dataset
    DS_RD_SETTING.previous_workspace_path = str(Path(base_model).absolute())
    DS_RD_SETTING.enable_model_dump = True
    DS_RD_SETTING.debug_timeout = 3600 * 10
    DS_RD_SETTING.full_timeout = 3600 * 100
    rd_loop: DataScienceRDLoop = DataScienceRDLoop(DS_RD_SETTING)
    asyncio.run(rd_loop.run())


if __name__ == "__main__":
    fire.Fire(main)
