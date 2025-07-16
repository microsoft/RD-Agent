"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv
"""

from dotenv import load_dotenv

load_dotenv(".env")
# 1) Make sure it is at the beginning of the script so that it will load dotenv before initializing BaseSettings.
# 2) The ".env" argument is necessary to make sure it loads `.env` from the current directory.

import subprocess
from importlib.resources import path as rpath

import fire

from rdagent.app.data_science.loop import main as data_science
from rdagent.app.general_model.general_model import (
    extract_models_and_implement as general_model,
)
from rdagent.app.qlib_rd_loop.factor import main as fin_factor
from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
from rdagent.app.qlib_rd_loop.model import main as fin_model
from rdagent.app.qlib_rd_loop.quant import main as fin_quant
from rdagent.app.utils.health_check import health_check
from rdagent.app.utils.info import collect_info
from rdagent.log.mle_summary import grade_summary


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


def app():
    fire.Fire(
        {
            "fin_factor": fin_factor,
            "fin_factor_report": fin_factor_report,
            "fin_model": fin_model,
            "fin_quant": fin_quant,
            "general_model": general_model,
            "ui": ui,
            "health_check": health_check,
            "collect_info": collect_info,
            "data_science": data_science,
            "grade_summary": grade_summary,
            "server_ui": server_ui,
        }
    )
