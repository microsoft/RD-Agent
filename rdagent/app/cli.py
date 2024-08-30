"""
CLI entrance for all rdagent application.

This will 
- make rdagent a nice entry and
- autoamtically load dotenv
"""
import platform
import subprocess
import docker
import sys
from pathlib import Path

import pkg_resources
from setuptools_scm import get_version

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


def sys_info():
    """collect system related info"""
    method_list = [
        ["Name of current operating system: ", "system"],
        ["Processor architecture: ", "machine"],
        ["System, version, and hardware information: ", "platform"],
        ["Version number of the system: ", "version"],
    ]
    for method in method_list:
        logger.info(f"{method[0]}{getattr(platform, method[1])()}")
    return None


def python_info():
    """collect Python related info"""
    python_version = sys.version.replace("\n", " ")
    logger.info(f"Python version: {python_version}")
    return None


def docker_info():
    client = docker.from_env()
    images = client.images.list()
    logger.info(f"Image ID: {images[0].id}")
    logger.info(f"Tags: {images[0].tags}")
    logger.info(f"Created: {images[0].attrs['Created']}")
    logger.info(f"Size: {images[0].attrs['Size']} bytes")
    logger.info(f"Architecture: {images[0].attrs['Architecture']}")
    logger.info(f"OS: {images[0].attrs['Os']}")


def rdagent_info():
    """collect rdagent related info"""
    root_dir = Path(__file__).resolve().parent.parent.parent
    current_version = get_version(root=root_dir)
    logger.info(f"RD-Agent version: {current_version.split('+')[0]}")
    package_list = [
        "autodoc-pydantic",
        "coverage",
        "furo",
        "git-changelog",
        "mypy[reports]",
        "myst-parser",
        "pytest",
        "Sphinx",
        "sphinx-autobuild",
        "sphinx-click",
        "sphinx-togglebutton",
        "sphinx_rtd_theme",
        "black",
        "isort",
        "mypy",
        "ruff",
        "toml-sort",
        "types-PyYAML",
        "types-psutil",
        "types-tqdm",
        "build",
        "setuptools-scm",
        "twine",
        "wheel",
        "coverage",
        "pytest",
    ]
    for package in package_list:
        version = pkg_resources.get_distribution(package).version
        logger.info(f"{package} version: {version}")
    return None


def collect_info():
    """Prints information about the system and the installed packages."""
    sys_info()
    python_info()
    docker_info()
    rdagent_info()
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
