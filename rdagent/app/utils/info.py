import platform
import sys
from pathlib import Path
import docker
import pkg_resources
from setuptools_scm import get_version
from rdagent.log import rdagent_logger as logger


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
    last_image = images[0]
    if images:
        logger.info(f"Image ID of the last run is: {last_image.id}")
        logger.info(f"Tags of the last run is: {last_image.tags}")
        logger.info(f"Created of the last run is: {last_image.attrs['Created']}")
        logger.info(f"Size of the last run is: {last_image.attrs['Size']} bytes")
        logger.info(f"Architecture of the last run is: {last_image.attrs['Architecture']}")
        logger.info(f"OS of the last run is: {last_image.attrs['Os']}")
    else:
        logger.info(f"No images.")
    containers = client.containers.list(all=True)
    containers.sort(key=lambda c: c.attrs["Created"])
    last_container = containers[-1]
    if containers:
        logger.info(f"Container ID: {last_container.id}")
        logger.info(f"Container Name: {last_container.name}")
        logger.info(f"Container Status: {last_container.status}")
        logger.info(f"Image ID used by the container: {last_container.image.id}")
        logger.info(f"Image tag used by the container: {last_container.image.tags}")
        logger.info(f"Container port mapping: {last_container.ports}")
        logger.info(f"Container Label: {last_container.labels}")
        logger.info(f"Startup Commands: {' '.join(client.containers.get(last_container.id).attrs['Config']['Cmd'])}")
    else:
        logger.info(f"No run containers.")



def rdagent_info():
    """collect rdagent related info"""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
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
