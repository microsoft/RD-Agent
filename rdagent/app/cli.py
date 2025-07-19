"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv
"""

import sys

from dotenv import load_dotenv

load_dotenv(".env")
# 1) Make sure it is at the beginning of the script so that it will load dotenv before initializing BaseSettings.
# 2) The ".env" argument is necessary to make sure it loads `.env` from the current directory.

import subprocess
from importlib.resources import path as rpath

import fire
import typer

from rdagent.app.data_science.loop import main as data_science_main
from rdagent.app.general_model.general_model import (
    extract_models_and_implement as general_model,
)
from rdagent.app.qlib_rd_loop.factor import main as fin_factor_main
from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
from rdagent.app.qlib_rd_loop.model import main as fin_model
from rdagent.app.qlib_rd_loop.quant import main as fin_quant
from rdagent.app.utils.health_check import health_check
from rdagent.app.utils.info import collect_info
from rdagent.log.mle_summary import grade_summary as grade_summary_main

app = typer.Typer()


def ui(port=19899, log_dir="", debug=False, data_science=False):
    """
    start web app to show the log traces.
    """
    if data_science:
        with rpath("rdagent.log.ui", "dsapp.py") as app_path:
            cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
            subprocess.run(cmds)
        return
    with rpath("rdagent.log.ui", "app.py") as app_path:
        cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
        if log_dir or debug:
            cmds.append("--")
        if log_dir:
            cmds.append(f"--log_dir={log_dir}")
        if debug:
            cmds.append("--debug")
        subprocess.run(cmds)


def server_ui(port=19899):
    """
    start web app to show the log traces in real time
    """
    subprocess.run(["python", "rdagent/log/server/app.py", f"--port={port}"])


def _run_fire(func, args: list[str]):
    sys.argv = [func.__name__] + args
    fire.Fire(func)


@app.command(name="grade_summary", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def grade_summary(ctx: typer.Context):
    """
    Generate test scores for log traces in the log folder and save the summary.

    Args:
        - log_folder (str | Path): log folder
    """
    _run_fire(grade_summary_main, list(ctx.args))


@app.command(name="data_science", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def data_science(ctx: typer.Context):
    """
    Auto R&D Evolving loop for data science.

    Args:
        - path (str | None): A path like `$LOG_PATH/__session__/1/0_propose`. This indicates that we restore the state after finishing step 0 in loop 1. Default is None.
        - checkout (bool | str | Path): Used only when a path is provided. Can be True, False, or a path. Default is True.
            - If True, the new loop will use the existing folder and clear logs for sessions after the one corresponding to the given path.
            - If False, the new loop will use the existing folder but keep the logs for sessions after the one corresponding to the given path.
            - If a path (or a str like Path) is provided, the new loop will be saved to that path, leaving the original path unchanged.
        - step_n (int | None): Number of steps to run; if None, the process will run indefinitely until an error or KeyboardInterrupt occurs. Default is None.
        - loop_n (int | None): Number of loops to run; if None, the process will run indefinitely until an error or KeyboardInterrupt occurs. Default is None.
            - If the current loop is incomplete, it will be counted as the first loop for completion.
            - If both step_n and loop_n are provided, the process will stop as soon as either condition is met.
        - competition (str): Competition name.  Default is bms-molecular-translation.
        - replace_timer (bool): If a session is loaded, determines whether to replace the timer with session.timer. Default is True.
        - exp_gen_cls (str | None): When there are different stages, the exp_gen can be replaced with the new proposal. Default is None.
    """
    _run_fire(data_science_main, list(ctx.args))


@app.command(name="fin_factor", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def fin_factor(ctx: typer.Context):
    """
    Auto R&D Evolving loop for fintech factors.

    path: str | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool | str | Path = True,

    Args:
        - path (str | None = None): A path like `$LOG_PATH/__session__/1/0_propose`. This indicates that we restore the state after finishing step 0 in loop 1. Default is None.
        - step_n (int | None = None): Number of steps to run; if None, the process will run indefinitely until an error or KeyboardInterrupt occurs. Default is None.
        - loop_n (int | None = None): Number of loops to run; if None, the process will run indefinitely until an error or KeyboardInterrupt occurs. Default is None.
        - all_duration (str | None): Maximum duration to run, in format accepted by the timer
        - checkout (bool | str | Path): Used only when a path is provided. Can be True, False, or a path. Default is True.
            - If True, the new loop will use the existing folder and clear logs for sessions after the one corresponding to the given path.
            - If False, the new loop will use the existing folder but keep the logs for sessions after the one corresponding to the given path.
            - If a path (or a str like Path) is provided, the new loop will be saved to that path, leaving the original path unchanged.
    """
    _run_fire(fin_factor_main, list(ctx.args))


app.command(name="fin_factor_report")(fin_factor_report)
app.command(name="fin_model")(fin_model)
app.command(name="fin_quant")(fin_quant)
app.command(name="general_model")(general_model)
app.command(name="ui")(ui)
app.command(name="health_check")(health_check)
app.command(name="collect_info")(collect_info)
app.command(name="server_ui")(server_ui)


if __name__ == "__main__":
    app()
