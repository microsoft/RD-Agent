from pathlib import Path

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.log.storage import FileStorage, WebStorage
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


def main(
    path: str | None = None,
    output_path: str | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    competition="bms-molecular-translation",
    do_truncate=True,
    timeout=None,
    replace_timer=True,
    exp_gen_cls: str | None = None,
):
    """

    Parameters
    ----------
    path :
        path like `$LOG_PATH/__session__/1/0_propose`. It indicates that we restore the state that after finish the step 0 in loop 1
    output_path :
        path like `$LOG_PATH`. It indicates that where we want to save our session and log information.
    step_n :
        How many steps to run; if None, it will run forever until error or KeyboardInterrupt
    loop_n :
        How many loops to run; if None, it will run forever until error or KeyboardInterrupt
        - if current loop is incomplete, it will be counted as the first loop for completion.
        - if both step_n and loop_n are provided, the process will stop as soon as either condition is met.
    competition :
    do_truncate :
        If set to True, the logger will truncate the future log messages by calling `logger.storage.truncate`.
    replace_timer :
        If session is loaded, should we replace the timer with session.timer
    exp_gen_cls :
        When we have different stages, we can replace the exp_gen with the new proposal


    Auto R&D Evolving loop for models in a Kaggle scenario.
    You can continue running session by
    .. code-block:: bash
        dotenv run -- python rdagent/app/data_science/loop.py [--competition titanic] $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional parameter
        rdagent kaggle --competition playground-series-s4e8  # You are encouraged to use this one.
    """
    if competition is not None:
        DS_RD_SETTING.competition = competition

    if not DS_RD_SETTING.competition:
        logger.error("Please specify competition name.")

    if path is None:
        logger.storages.append(FileStorage(RD_AGENT_SETTINGS.log_trace_path))
        kaggle_loop = DataScienceRDLoop(DS_RD_SETTING)
        trace_id = RD_AGENT_SETTINGS.log_trace_path
    else:
        kaggle_loop: DataScienceRDLoop = DataScienceRDLoop.load(path, output_path, replace_timer)

        existing_storage = FileStorage(Path(path).parent.parent.parent)
        if do_truncate:
            max_loop = max(kaggle_loop.loop_trace.keys())
            existing_storage.truncate(time=kaggle_loop.loop_trace[max_loop][-1].end)
        logger.storages.append(existing_storage)
        trace_id = str(existing_storage.path)

    if RD_AGENT_SETTINGS.ui_server_port:
        logger.storages.append(WebStorage(RD_AGENT_SETTINGS.ui_server_port, id=trace_id))

    # replace exp_gen if we have new class
    if exp_gen_cls is not None:
        kaggle_loop.exp_gen = import_class(exp_gen_cls)(kaggle_loop.exp_gen.scen)

    kaggle_loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout)


if __name__ == "__main__":
    fire.Fire(main)
