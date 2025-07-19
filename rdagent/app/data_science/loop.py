import asyncio
from pathlib import Path
from typing import Optional

import fire
import typer
from typing_extensions import Annotated

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


def main(
    path: Optional[str] = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    checkout_path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
    competition="bms-molecular-translation",
    replace_timer=True,
    exp_gen_cls: Optional[str] = None,
):
    """

    Parameters
    ----------
    path :
        A path like `$LOG_PATH/__session__/1/0_propose`. This indicates that we restore the state after finishing step 0 in loop 1.
    checkout :
        Used to control the log session path. Boolean type, default is True.
        - If True, the new loop will use the existing folder and clear logs for sessions after the one corresponding to the given path.
        - If False, the new loop will use the existing folder but keep the logs for sessions after the one corresponding to the given path.
    checkout_path:
        If a checkout_path (or a str like Path) is provided, the new loop will be saved to that path, leaving the original path unchanged.
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
    if not checkout_path is None:
        checkout = Path(checkout_path)

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
