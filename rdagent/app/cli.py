"""
CLI entrance for all rdagent application.

This will 
- make rdagent a nice entry and
- autoamtically load dotenv
"""
import platform
import subprocess
import sys
from pathlib import Path

import pkg_resources
from setuptools_scm import get_version

Path(__file__).resolve().parent.parent
from importlib.resources import path as rpath

import fire
from dotenv import load_dotenv

from rdagent.app.data_mining.model import main as med_model
from rdagent.app.general_model.general_model import (
    extract_models_and_implement as general_model,
)
from rdagent.app.qlib_rd_loop.factor import main as fin_factor
from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
from rdagent.app.qlib_rd_loop.model import main as fin_model
from rdagent.log import rdagent_logger as logger

load_dotenv()


def ui(port=80, log_dir="", debug=False):
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

def find_project_root():
    """Find the root directory of the project."""
    current_path = Path.cwd()
    marker = 'pyproject.toml'
    current_path = current_path.resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    return None

def collect_info():
    """Prints information about the system and the installed packages."""
    method_list = [
        ["Name of current operating system: ", "system"],
        ["Processor architecture: ", "machine"],
        ["System, version, and hardware information: ", "platform"],
        ["Version number of the system: ", "version"],
    ]
    for method in method_list:
        logger.info(f"{method[0]}{getattr(platform, method[1])()}")
    python_version = sys.version.replace("\n", " ")
    logger.info(f"Python version: {python_version}")
    root_dir = find_project_root()
    current_version = get_version(root = root_dir)
    logger.info(f"RD-Agent version: {current_version.split('+')[0]}")
    file_name = ".".join(python_version.split()[0].split(".")[:2]) + ".txt"
    file_path = root_dir / "constraints" / file_name
    if not file_path.exists():
        logger.error(f"{file_path} doesn't exist. It could be that your python version is incorrect.")
        return None
    with open(file_path) as f:
        package_list = f.readlines()
    package_name_list = [pkg.split('==')[0] for pkg in package_list]
    for package in package_name_list:
        version = pkg_resources.get_distribution(package).version
        logger.info(f"{package} version: {version}")
    return None


def app():
    fire.Fire(
        {
            "fin_factor": fin_factor,
            "fin_factor_report": fin_factor_report,
            "fin_model": fin_model,
            "med_model": med_model,
            "general_model": general_model,
            "ui": ui,
            "collect_info": collect_info,
        }
    )
