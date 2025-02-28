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
import time
from importlib.resources import path as rpath

import fire
import requests

from rdagent.app.data_mining.model import main as med_model
from rdagent.app.general_model.general_model import (
    extract_models_and_implement as general_model,
)
from rdagent.app.kaggle.loop import main as kaggle_main
from rdagent.app.qlib_rd_loop.factor import main as fin_factor
from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
from rdagent.app.qlib_rd_loop.model import main as fin_model
from rdagent.app.utils.health_check import health_check
from rdagent.app.utils.info import collect_info
from rdagent.log import rdagent_logger as logger


def ui(port=19899, log_dir="", debug=False):
    """
    start web app to show the log traces.
    """
    with rpath("rdagent.log.ui", "app.py") as app_path:
        cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
        if log_dir or debug:
            cmds.append("--")
        if log_dir:
            cmds.append(f"--log_dir={log_dir}")
        if debug:
            cmds.append("--debug")
        subprocess.run(cmds)


def start_flask_server():
    """
    Start the Flask server and make sure it is running in the background without affecting the main process.
    """
    flask_process = subprocess.Popen(
        ["python", "rdagent/log/server/app.py"],
        # Hide Output
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # Standard Output
        # stdout=sys.stdout,
        # stderr=sys.stderr,
    )
    flask_url = "http://127.0.0.1:19899"
    for _ in range(6):
        try:
            response = requests.get(flask_url, timeout=1)
            if response.status_code == 200:
                logger.info(f"The Flask server started successfully. Running on: {flask_url}")
                return flask_process
        except requests.exceptions.RequestException:
            time.sleep(1)
    logger.error("❌ Flask server startup failed, please check manually!")


def app():
    flask_process = start_flask_server()
    try:
        fire.Fire(
            {
                "fin_factor": fin_factor,
                "fin_factor_report": fin_factor_report,
                "fin_model": fin_model,
                "med_model": med_model,
                "general_model": general_model,
                "ui": ui,
                "health_check": health_check,
                "collect_info": collect_info,
                "kaggle": kaggle_main,
            }
        )
    finally:
        # Terminate the Flask process to prevent it from running after `app()` exits.
        flask_process.terminate()
