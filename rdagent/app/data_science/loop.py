import asyncio
from pathlib import Path

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


def main(
    path: str | None = None,
    checkout: bool | str | Path = True,
    step_n: int | None = None,
    loop_n: int | None = None,
    timeout: str | None = None,
    competition="bms-molecular-translation",
    replace_timer=True,
    exp_gen_cls: str | None = None,
):
    """

    Parameters
    ----------
    path :
        A path like `$LOG_PATH/__session__/1/0_propose`. This indicates that we restore the state after finishing step 0 in loop 1.
    checkout :
        Used only when a path is provided.
        Can be True, False, or a path.
        Default is True.
        - If True, the new loop will use the existing folder and clear logs for sessions after the one corresponding to the given path.
        - If False, the new loop will use the existing folder but keep the logs for sessions after the one corresponding to the given path.
        - If a path (or a str like Path) is provided, the new loop will be saved to that path, leaving the original path unchanged.
    step_n :
        Number of steps to run; if None, the process will run indefinitely until an error or KeyboardInterrupt occurs.
    loop_n :
        Number of loops to run; if None, the process will run indefinitely until an error or KeyboardInterrupt occurs.
        - If the current loop is incomplete, it will be counted as the first loop for completion.
        - If both step_n and loop_n are provided, the process will stop as soon as either condition is met.
    competition :
        Competition name.
    replace_timer :
        If a session is loaded, determines whether to replace the timer with session.timer.
    exp_gen_cls :
        When there are different stages, the exp_gen can be replaced with the new proposal.


    Auto R&D Evolving loop for models in a Kaggle scenario.
    You can continue running a session by using the command:
    .. code-block:: bash
        dotenv run -- python rdagent/app/data_science/loop.py [--competition titanic] $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is an optional parameter
        rdagent kaggle --competition playground-series-s4e8  # This command is recommended.
    """
    if competition is not None:
        DS_RD_SETTING.competition = competition

    if not DS_RD_SETTING.competition:
        logger.error("Please specify competition name.")

    if path is None:
        kaggle_loop = DataScienceRDLoop(DS_RD_SETTING)
    else:
        kaggle_loop: DataScienceRDLoop = DataScienceRDLoop.load(path, checkout=checkout, replace_timer=replace_timer)

    # replace exp_gen if we have new class
    if exp_gen_cls is not None:
        kaggle_loop.exp_gen = import_class(exp_gen_cls)(kaggle_loop.exp_gen.scen)

    asyncio.run(kaggle_loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))


if __name__ == "__main__":
    fire.Fire(main)
