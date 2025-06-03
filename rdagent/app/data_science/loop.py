import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop


def main(
    path=None,
    output_path=None,
    step_n=None,
    loop_n=None,
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
        kaggle_loop = DataScienceRDLoop(DS_RD_SETTING)
    else:
        kaggle_loop = DataScienceRDLoop.load(path, output_path, do_truncate, replace_timer)

    # replace exp_gen if we have new class
    if exp_gen_cls is not None:
        kaggle_loop.exp_gen = import_class(exp_gen_cls)(kaggle_loop.exp_gen.scen)

    kaggle_loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout)


if __name__ == "__main__":
    fire.Fire(main)
