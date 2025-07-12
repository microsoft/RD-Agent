"""
The motivation of the utils is for environment management

Tries to create uniform environment for the agent to run;
- All the code and data is expected included in one folder
"""

# TODO: move the scenario specific docker env into other folders.

import contextlib
import json
import os
import pickle
import re
import select
import shutil
import subprocess
import time
import uuid
import zipfile
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generator, Generic, Mapping, Optional, TypeVar, cast

import docker  # type: ignore[import-untyped]
import docker.models  # type: ignore[import-untyped]
import docker.models.containers  # type: ignore[import-untyped]
import docker.types  # type: ignore[import-untyped]
from pydantic import BaseModel, model_validator
from pydantic_settings import SettingsConfigDict
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm

from rdagent.core.conf import ExtendedBaseSettings
from rdagent.core.experiment import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


def cleanup_container(container: docker.models.containers.Container | None, context: str = "") -> None:  # type: ignore[no-any-unimported]
    """
    Shared helper function to clean up a Docker container.
    Always stops the container before removing it.

    Parameters
    ----------
    container : docker container object or None
        The container to clean up, or None if no container to clean up
    context : str
        Additional context for logging (e.g., "health check", "GPU test")
    """
    if container is not None:
        try:
            # Always stop first - stop() doesn't raise error if already stopped
            container.stop()
            container.remove()
        except Exception as cleanup_error:
            # Log cleanup error but don't mask the original exception
            context_str = f" {context}" if context else ""
            logger.warning(f"Failed to cleanup{context_str} container {container.id}: {cleanup_error}")


# Normalize all bind paths in volumes to absolute paths using the workspace (working_dir).
def normalize_volumes(vols: dict[str, str | dict[str, str]], working_dir: str) -> dict:
    abs_vols: dict[str, str | dict[str, str]] = {}

    def to_abs(path: str) -> str:
        # Converts a relative path to an absolute path using the workspace (working_dir).
        return os.path.abspath(os.path.join(working_dir, path)) if not os.path.isabs(path) else path

    for lp, vinfo in vols.items():
        # Support both:
        # 1. {'host_path': {'bind': 'container_path', ...}}
        # 2. {'host_path': 'container_path'}
        if isinstance(vinfo, dict):
            # abs_vols = cast(dict[str, dict[str, str]], abs_vols)
            vinfo = vinfo.copy()
            vinfo["bind"] = to_abs(vinfo["bind"])
            abs_vols[lp] = vinfo
        else:
            # abs_vols = cast(dict[str, str], abs_vols)
            abs_vols[lp] = to_abs(vinfo)
    return abs_vols


def pull_image_with_progress(image: str) -> None:
    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    pull_logs = client.pull(image, stream=True, decode=True)
    progress_bars = {}

    for log in pull_logs:
        if "id" in log and log.get("progressDetail"):
            layer_id = log["id"]
            progress_detail = log["progressDetail"]
            current = progress_detail.get("current", 0)
            total = progress_detail.get("total", 0)

            if total:
                if layer_id not in progress_bars:
                    progress_bars[layer_id] = tqdm(total=total, desc=f"Layer {layer_id}", unit="B", unit_scale=True)
                progress_bars[layer_id].n = current
                progress_bars[layer_id].refresh()

        elif "status" in log:
            print(log["status"])

    for pb in progress_bars.values():
        pb.close()


class EnvConf(ExtendedBaseSettings):
    default_entry: str
    extra_volumes: dict = {}
    running_timeout_period: int | None = 3600  # 10 minutes
    # helper settings to support transparent;
    enable_cache: bool = True
    retry_count: int = 5  # retry count for the docker run
    retry_wait_seconds: int = 10  # retry wait seconds for the docker run

    model_config = SettingsConfigDict(
        # TODO: add prefix ....
        env_parse_none_str="None",  # Nthis is the key to accept `RUNNING_TIMEOUT_PERIOD=None`
    )


ASpecificEnvConf = TypeVar("ASpecificEnvConf", bound=EnvConf)


@dataclass
class EnvResult:
    """
    The result of running the environment.
    It contains the stdout, the exit code, and the running time in seconds.
    """

    stdout: str
    exit_code: int
    running_time: float


class Env(Generic[ASpecificEnvConf]):
    """
    We use BaseModel as the setting due to the features it provides
    - It provides base typing and checking features.
    - loading and dumping the information will be easier: for example, we can use package like `pydantic-yaml`
    """

    conf: ASpecificEnvConf  # different env have different conf.

    def __init__(self, conf: ASpecificEnvConf):
        self.conf = conf

    def zip_a_folder_into_a_file(self, folder_path: str, zip_file_path: str) -> None:
        """
        Zip a folder into a file, use zipfile instead of subprocess
        """
        with zipfile.ZipFile(zip_file_path, "w") as z:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))

    def unzip_a_file_into_a_folder(self, zip_file_path: str, folder_path: str) -> None:
        """
        Unzip a file into a folder, use zipfile instead of subprocess
        """
        # Clear folder_path before extracting
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

        with zipfile.ZipFile(zip_file_path, "r") as z:
            z.extractall(folder_path)

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Prepare for the environment based on it's configure
        """

    def check_output(
        self, entry: str | None = None, local_path: str = ".", env: dict | None = None, **kwargs: dict
    ) -> str:
        """
        Run the folder under the environment.

        Parameters
        ----------
        entry : str | None
            We may we the entry point when we run it.
            For example, we may have different entries when we run and summarize the project.
        local_path : str | None
            the local path (to project, mainly for code) will be mounted into the docker
            Here are some examples for a None local path
            - for example, run docker for updating the data in the extra_volumes.
            - simply run the image. The results are produced by output or network
        env : dict | None
            Run the code with your specific environment.

        Returns
        -------
            the stdout
        """
        result = self.run(entry=entry, local_path=local_path, env=env, **kwargs)
        return result.stdout

    def __run_with_retry(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
    ) -> EnvResult:
        for retry_index in range(self.conf.retry_count + 1):
            try:
                start = time.time()
                log_output, return_code = self._run(
                    entry,
                    local_path,
                    env,
                    running_extra_volume=running_extra_volume,
                )
                end = time.time()
                logger.info(f"Running time: {end - start} seconds")
                if self.conf.running_timeout_period is not None and end - start + 1 >= self.conf.running_timeout_period:
                    logger.warning(
                        f"The running time exceeds {self.conf.running_timeout_period} seconds, so the process is killed."
                    )
                    log_output += f"\n\nThe running time exceeds {self.conf.running_timeout_period} seconds, so the process is killed."
                log_output += f"\nTotal running time: {end - start:.3f} seconds."
                return EnvResult(log_output, return_code, end - start)
            except Exception as e:
                if retry_index == self.conf.retry_count:
                    raise
                logger.warning(
                    f"Error while running the container: {e}, current try index: {retry_index + 1}, {self.conf.retry_count - retry_index - 1} retries left."
                )
                time.sleep(self.conf.retry_wait_seconds)
        raise RuntimeError  # for passing CI

    def run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        **kwargs: dict,
    ) -> EnvResult:
        """
        Run the folder under the environment and return the stdout, exit code, and running time.

        Parameters
        ----------
        entry : str | None
            We may we the entry point when we run it.
            For example, we may have different entries when we run and summarize the project.
        local_path : str | None
            the local path (to project, mainly for code) will be mounted into the docker
            Here are some examples for a None local path
            - for example, run docker for updating the data in the extra_volumes.
            - simply run the image. The results are produced by output or network
        env : dict | None
            Run the code with your specific environment.

        Returns
        -------
            EnvResult: An object containing the stdout, the exit code, and the running time in seconds.
        """
        running_extra_volume = kwargs.get("running_extra_volume", {})
        if entry is None:
            entry = self.conf.default_entry

        if "|" in entry:
            logger.warning(
                "You are using a command with a shell pipeline (i.e., '|'). "
                "The exit code ($exit_code) will reflect the result of "
                "the last command in the pipeline.",
            )

        # FIXME: the input path and cache path is hard coded here.
        # We don't want to change the content in input and cache path.
        # Otherwise, it may produce large amount of warnings.
        def _get_chmod_cmd(workspace_path: str) -> str:
            def _get_path_stem(path: str) -> str | None:
                # If the input path is relative, keep only the first component
                p = Path(path)
                if not p.is_absolute() and p.parts:
                    return p.parts[0]
                return None

            chmod_cmd = f"chmod -R 777 $(find {workspace_path} -mindepth 1 -maxdepth 1"
            for name in [
                _get_path_stem(T("scenarios.data_science.share:scen.cache_path").r()),
                _get_path_stem(T("scenarios.data_science.share:scen.input_path").r()),
            ]:
                chmod_cmd += f" ! -name {name}"
            chmod_cmd += ")"
            return chmod_cmd

        if self.conf.running_timeout_period is None:
            timeout_cmd = entry
        else:
            timeout_cmd = f"timeout --kill-after=10 {self.conf.running_timeout_period} {entry}"
        entry_add_timeout = (
            f"/bin/sh -c '"  # start of the sh command
            + f"{timeout_cmd}; entry_exit_code=$?; "
            + (
                f"{_get_chmod_cmd(self.conf.mount_path)}; "
                # We don't have to change the permission of the cache and input folder to remove it
                # + f"if [ -d {self.conf.mount_path}/cache ]; then chmod 777 {self.conf.mount_path}/cache; fi; " +
                #     f"if [ -d {self.conf.mount_path}/input ]; then chmod 777 {self.conf.mount_path}/input; fi; "
                if isinstance(self.conf, DockerConf)
                else ""
            )
            + "exit $entry_exit_code"
            + "'"  # end of the sh command
        )

        if self.conf.enable_cache:
            result = self.cached_run(entry_add_timeout, local_path, env, running_extra_volume)
        else:
            result = self.__run_with_retry(
                entry_add_timeout,
                local_path,
                env,
                running_extra_volume,
            )

        return result

    def cached_run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
    ) -> EnvResult:
        """
        Run the folder under the environment.
        Will cache the output and the folder diff for next round of running.
        Use the python codes and the parameters(entry, running_extra_volume) as key to hash the input.
        """
        target_folder = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str) / f"utils.env.run"
        target_folder.mkdir(parents=True, exist_ok=True)

        # we must add the information of data (beyond code) into the key.
        # Otherwise, all commands operating on data will become invalid (e.g. rm -r submission.csv)
        # So we recursively walk in the folder and add the sorted relative filename list as part of the key.
        # data_key = []
        # for path in Path(local_path).rglob("*"):
        #     p = str(path.relative_to(Path(local_path)))
        #     if p.startswith("__pycache__"):
        #         continue
        #     data_key.append(p)
        # data_key = sorted(data_key)

        key = md5_hash(
            json.dumps(
                [
                    [str(path.relative_to(Path(local_path))), path.read_text()]
                    for path in sorted(list(Path(local_path).rglob("*.py")) + list(Path(local_path).rglob("*.csv")))
                ]
            )
            + json.dumps({"entry": entry, "running_extra_volume": dict(running_extra_volume)})
            + json.dumps({"extra_volumes": self.conf.extra_volumes})
            # + json.dumps(data_key)
        )
        if Path(target_folder / f"{key}.pkl").exists() and Path(target_folder / f"{key}.zip").exists():
            with open(target_folder / f"{key}.pkl", "rb") as f:
                ret = pickle.load(f)
            self.unzip_a_file_into_a_folder(str(target_folder / f"{key}.zip"), local_path)
        else:
            ret = self.__run_with_retry(entry, local_path, env, running_extra_volume)
            with open(target_folder / f"{key}.pkl", "wb") as f:
                pickle.dump(ret, f)
            self.zip_a_folder_into_a_file(local_path, str(target_folder / f"{key}.zip"))
        return cast(EnvResult, ret)

    @abstractmethod
    def _run(
        self,
        entry: str | None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        """
        Execute the specified entry point within the given environment and local path.

        Parameters
        ----------
        entry : str | None
            The entry point to execute. If None, defaults to the configured entry.
        local_path : str
            The local directory path where the execution should occur.
        env : dict | None
            Environment variables to set during execution.
        kwargs : dict
            Additional keyword arguments for execution customization.

        Returns
        -------
        tuple[str, int]
            A tuple containing the standard output and the exit code.
        """
        pass

    def dump_python_code_run_and_get_results(
        self,
        code: str,
        dump_file_names: list[str],
        local_path: str,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        code_dump_file_py_name: Optional[str] = None,
    ) -> tuple[str, list]:
        """
        Dump the code into the local path and run the code.
        """
        random_file_name = f"{uuid.uuid4()}.py" if code_dump_file_py_name is None else f"{code_dump_file_py_name}.py"
        with open(os.path.join(local_path, random_file_name), "w") as f:
            f.write(code)
        entry = f"python {random_file_name}"
        log_output = self.check_output(entry, local_path, env, running_extra_volume=dict(running_extra_volume))
        results = []
        os.remove(os.path.join(local_path, random_file_name))
        for name in dump_file_names:
            if os.path.exists(os.path.join(local_path, f"{name}")):
                results.append(pickle.load(open(os.path.join(local_path, f"{name}"), "rb")))
                os.remove(os.path.join(local_path, f"{name}"))
            else:
                return log_output, []
        return log_output, results


# class EnvWithCache
#

## Local Environment -----


class LocalConf(EnvConf):
    bin_path: str = ""
    """path like <path1>:<path2>:<path3>, which will be prepend to bin path."""

    retry_count: int = 0  # retry count for; run `retry_count + 1` times
    live_output: bool = False


ASpecificLocalConf = TypeVar("ASpecificLocalConf", bound=LocalConf)


class LocalEnv(Env[ASpecificLocalConf]):
    """
    Sometimes local environment may be more convenient for testing
    """

    def prepare(self) -> None: ...

    def _run(
        self,
        entry: str | None = None,
        local_path: str | None = None,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: dict,
    ) -> tuple[str, int]:

        # Handle volume links
        volumes = {}
        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = T("scenarios.data_science.share:scen.cache_path").r()
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp

        assert local_path is not None, "local_path should not be None"
        volumes = normalize_volumes(volumes, local_path)

        @contextlib.contextmanager
        def _symlink_ctx(vol_map: Mapping[str, str]) -> Generator[None, None, None]:
            created_links: list[Path] = []
            try:
                for real, link in vol_map.items():
                    link_path = Path(link)
                    real_path = Path(real)
                    if not link_path.parent.exists():
                        link_path.parent.mkdir(parents=True, exist_ok=True)
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    link_path.symlink_to(real_path)
                    created_links.append(link_path)
                yield
            finally:
                for p in created_links:
                    try:
                        if p.is_symlink() or p.exists():
                            p.unlink()
                    except FileNotFoundError:
                        pass

        with _symlink_ctx(volumes):
            # Setup environment
            if env is None:
                env = {}
            path = [*self.conf.bin_path.split(":"), "/bin/", "/usr/bin/", *env.get("PATH", "").split(":")]
            env["PATH"] = ":".join(path)

            if entry is None:
                entry = self.conf.default_entry

            print(Rule("[bold green]LocalEnv Logs Begin[/bold green]", style="dark_orange"))
            table = Table(title="Run Info", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="bold magenta")
            table.add_row("Entry", entry)
            table.add_row("Local Path", local_path or "")
            table.add_row("Env", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("Volumes", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)

            cwd = Path(local_path).resolve() if local_path else None
            env = {k: str(v) if isinstance(v, int) else v for k, v in env.items()}

            process = subprocess.Popen(
                entry,
                cwd=cwd,
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Setup polling
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("The subprocess did not correctly create stdout/stderr pipes")

            if self.conf.live_output:
                stdout_fd = process.stdout.fileno()
                stderr_fd = process.stderr.fileno()

                poller = select.poll()
                poller.register(stdout_fd, select.POLLIN)
                poller.register(stderr_fd, select.POLLIN)

                combined_output = ""
                while True:
                    if process.poll() is not None:
                        break
                    events = poller.poll(100)
                    for fd, event in events:
                        if event & select.POLLIN:
                            if fd == stdout_fd:
                                while True:
                                    output = process.stdout.readline()
                                    if output == "":
                                        break
                                    Console().print(output.strip(), markup=False)
                                    combined_output += output
                            elif fd == stderr_fd:
                                while True:
                                    error = process.stderr.readline()
                                    if error == "":
                                        break
                                    Console().print(error.strip(), markup=False)
                                    combined_output += error

                # Capture any final output
                remaining_output, remaining_error = process.communicate()
                if remaining_output:
                    Console().print(remaining_output.strip(), markup=False)
                    combined_output += remaining_output
                if remaining_error:
                    Console().print(remaining_error.strip(), markup=False)
                    combined_output += remaining_error
            else:
                # Sacrifice real-time output to avoid possible standard I/O hangs
                out, err = process.communicate()
                Console().print(out, end="", markup=False)
                Console().print(err, end="", markup=False)
                combined_output = out + err

            return_code = process.returncode
            print(Rule("[bold green]LocalEnv Logs End[/bold green]", style="dark_orange"))

            return combined_output, return_code


class CondaConf(LocalConf):
    conda_env_name: str
    default_entry: str = "python main.py"

    @model_validator(mode="after")
    def change_bin_path(self, **data: Any) -> "CondaConf":
        conda_path_result = subprocess.run(
            f"conda run -n {self.conda_env_name} --no-capture-output env | grep '^PATH='",
            capture_output=True,
            text=True,
            shell=True,
        )
        self.bin_path = conda_path_result.stdout.strip().split("=")[1] if conda_path_result.returncode == 0 else ""
        return self


class MLECondaConf(CondaConf):
    enable_cache: bool = False  # aligning with the docker settings.


## Docker Environment -----
class DockerConf(EnvConf):
    build_from_dockerfile: bool = False
    dockerfile_folder_path: Optional[Path] = (
        None  # the path to the dockerfile optional path provided when build_from_dockerfile is False
    )
    image: str  # the image you want to build
    mount_path: str  # the path in the docker image to mount the folder
    default_entry: str  # the entry point of the image

    extra_volumes: dict = {}
    """It accept a dict of volumes, which can be either
    {<host_path>: <container_path>} or
    {<host_path>: {"bind": <container_path>, "mode": <mode, ro/rw/default is extra_volume_mode>}}
    """
    extra_volume_mode: str = "ro"  # by default. only the mount_path should be writable, others are changed to read-only
    # Sometime, we need maintain some extra data for the workspace.
    # And the extra data may be shared and the downloading can be time consuming.
    # So we just want to download it once.
    network: str | None = "bridge"  # the network mode for the docker
    shm_size: str | None = None
    enable_gpu: bool = True  # because we will automatically disable GPU if not available. So we enable it by default.
    mem_limit: str | None = "48g"  # Add memory limit attribute
    cpu_count: int | None = None  # Add CPU limit attribute

    running_timeout_period: int | None = 3600  # 1 hour

    enable_cache: bool = True  # enable the cache mechanism

    retry_count: int = 5  # retry count for the docker run
    retry_wait_seconds: int = 10  # retry wait seconds for the docker run


class QlibCondaConf(CondaConf):
    conda_env_name: str = "rdagent4qlib"
    enable_cache: bool = False
    default_entry: str = "qrun conf.yaml"
    # extra_volumes: dict = {str(Path("~/.qlib/").expanduser().resolve().absolute()): "/root/.qlib/"}


class QlibCondaEnv(LocalEnv[QlibCondaConf]):
    def prepare(self) -> None:
        """Prepare the conda environment if not already created."""
        try:
            envs = subprocess.run("conda env list", capture_output=True, text=True, shell=True)
            if self.conf.conda_env_name not in envs.stdout:
                print(f"[yellow]Conda env '{self.conf.conda_env_name}' not found, creating...[/yellow]")
                subprocess.check_call(
                    f"conda create -y -n {self.conf.conda_env_name} python=3.10",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install --upgrade pip cython",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install git+https://github.com/microsoft/qlib.git@3e72593b8c985f01979bebcf646658002ac43b00",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install catboost xgboost scipy==1.11.4 tables torch",
                    shell=True,
                )
        except Exception as e:
            print(f"[red]Failed to prepare conda env: {e}[/red]")


class QlibDockerConf(DockerConf):
    model_config = SettingsConfigDict(
        env_prefix="QLIB_DOCKER_",
        env_parse_none_str="None",  # Nthis is the key to accept `RUNNING_TIMEOUT_PERIOD=None`
    )

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "qlib" / "docker"
    image: str = "local_qlib:latest"
    mount_path: str = "/workspace/qlib_workspace/"
    default_entry: str = "qrun conf.yaml"
    extra_volumes: dict = {
        str(Path("~/.qlib/").expanduser().resolve().absolute()): {"bind": "/root/.qlib/", "mode": "rw"}
    }
    shm_size: str | None = "16g"
    enable_gpu: bool = True
    enable_cache: bool = False


class KGDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="KG_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "kaggle_docker"
    image: str = "local_kg:latest"
    # image: str = "gcr.io/kaggle-gpu-images/python:latest"
    mount_path: str = "/workspace/kg_workspace/"
    default_entry: str = "python train.py"
    # extra_volumes: dict = {
    #     # TODO connect to the place where the data is stored
    #     Path("git_ignore_folder/data").resolve(): "/root/.data/"
    # }

    running_timeout_period: int | None = 600
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )


class DSDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="DS_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "DS_docker"
    image: str = "local_ds:latest"
    mount_path: str = "/kaggle/workspace"
    default_entry: str = "python main.py"

    running_timeout_period: int | None = 600
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )


class MLEBDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="MLEB_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "mle_bench_docker"
    image: str = "local_mle:latest"
    # image: str = "gcr.io/kaggle-gpu-images/python:latest"
    mount_path: str = "/workspace/data_folder/"
    default_entry: str = "mlebench prepare --all"
    # extra_volumes: dict = {
    #     # TODO connect to the place where the data is stored
    #     Path("git_ignore_folder/data").resolve(): "/root/.data/"
    # }
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )
    enable_cache: bool = False


# physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3
class DockerEnv(Env[DockerConf]):
    # TODO: Save the output into a specific file

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Download image if it doesn't exist
        """
        client = docker.from_env()
        if (
            self.conf.build_from_dockerfile
            and self.conf.dockerfile_folder_path is not None
            and self.conf.dockerfile_folder_path.exists()
        ):
            logger.info(f"Building the image from dockerfile: {self.conf.dockerfile_folder_path}")
            resp_stream = client.api.build(
                path=str(self.conf.dockerfile_folder_path), tag=self.conf.image, network_mode=self.conf.network
            )
            if isinstance(resp_stream, str):
                logger.info(resp_stream)
            with Progress(SpinnerColumn(), TextColumn("{task.description}")) as p:
                task = p.add_task("[cyan]Building image...")
                for part in resp_stream:
                    lines = part.decode("utf-8").split("\r\n")
                    for line in lines:
                        if line.strip():
                            status_dict = json.loads(line)
                            if "error" in status_dict:
                                p.update(task, description=f"[red]error: {status_dict['error']}")
                                raise docker.errors.BuildError(status_dict["error"], "")
                            if "stream" in status_dict:
                                p.update(task, description=status_dict["stream"])
            logger.info(f"Finished building the image from dockerfile: {self.conf.dockerfile_folder_path}")
        try:
            client.images.get(self.conf.image)
        except docker.errors.ImageNotFound:
            image_pull = client.api.pull(self.conf.image, stream=True, decode=True)
            current_status = ""
            layer_set = set()
            completed_layers = 0
            with Progress(TextColumn("{task.description}"), TextColumn("{task.fields[progress]}")) as sp:
                main_task = sp.add_task("[cyan]Pulling image...", progress="")
                status_task = sp.add_task("[bright_magenta]layer status", progress="")
                for line in image_pull:
                    if "error" in line:
                        sp.update(status_task, description=f"[red]error", progress=line["error"])
                        raise docker.errors.APIError(line["error"])

                    layer_id = line["id"]
                    status = line["status"]
                    p_text = line.get("progress", None)

                    if layer_id not in layer_set:
                        layer_set.add(layer_id)

                    if p_text:
                        current_status = p_text

                    if status == "Pull complete" or status == "Already exists":
                        completed_layers += 1

                    sp.update(main_task, progress=f"[green]{completed_layers}[white]/{len(layer_set)} layers completed")
                    sp.update(
                        status_task,
                        description=f"[bright_magenta]layer {layer_id} [yellow]{status}",
                        progress=current_status,
                    )
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while pulling the image: {e}")

    def _gpu_kwargs(self, client: docker.DockerClient) -> dict:  # type: ignore[no-any-unimported]
        """get gpu kwargs based on its availability"""
        if not self.conf.enable_gpu:
            return {}
        gpu_kwargs = {
            "device_requests": (
                [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if self.conf.enable_gpu else None
            ),
        }

        def get_image(image_name: str) -> None:
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                pull_image_with_progress(image_name)

        @wait_retry(5, 10)
        def _f() -> dict:
            container = None
            try:
                get_image(self.conf.image)
                container = client.containers.run(self.conf.image, "nvidia-smi", detach=True, **gpu_kwargs)
                # Wait for container to complete
                container.wait()
                logger.info("GPU Devices are available.")
            except docker.errors.APIError:
                return {}
            finally:
                cleanup_container(container, context="GPU test")
            return gpu_kwargs

        return _f()

    def _run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        if env is None:
            env = {}
        env["PYTHONWARNINGS"] = "ignore"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        client = docker.from_env()

        volumes = {}
        if local_path is not None:
            local_path = os.path.abspath(local_path)
            volumes[local_path] = {"bind": self.conf.mount_path, "mode": "rw"}

        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = {"bind": T("scenarios.data_science.share:scen.cache_path").r(), "mode": "rw"}
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}

        volumes = normalize_volumes(cast(dict[str, str | dict[str, str]], volumes), self.conf.mount_path)

        log_output = ""
        container: docker.models.containers.Container | None = None  # type: ignore[no-any-unimported]

        try:
            container = client.containers.run(
                image=self.conf.image,
                command=entry,
                volumes=volumes,
                environment=env,
                detach=True,
                working_dir=self.conf.mount_path,
                # auto_remove=True, # remove too fast might cause the logs not to be get
                network=self.conf.network,
                shm_size=self.conf.shm_size,
                mem_limit=self.conf.mem_limit,  # Set memory limit
                cpu_count=self.conf.cpu_count,  # Set CPU limit
                **self._gpu_kwargs(client),
            )
            assert container is not None  # Ensure container was created successfully
            logs = container.logs(stream=True)
            print(Rule("[bold green]Docker Logs Begin[/bold green]", style="dark_orange"))
            table = Table(title="Run Info", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="bold magenta")
            table.add_row("Image", self.conf.image)
            table.add_row("Container ID", container.id)
            table.add_row("Container Name", container.name)
            table.add_row("Entry", entry)
            table.add_row("Env", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("Volumes", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)
            for log in logs:
                decoded_log = log.strip().decode()
                Console().print(decoded_log, markup=False)
                log_output += decoded_log + "\n"
            exit_status = container.wait()["StatusCode"]
            print(Rule("[bold green]Docker Logs End[/bold green]", style="dark_orange"))
            return log_output, exit_status
        except docker.errors.ContainerError as e:
            raise RuntimeError(f"Error while running the container: {e}")
        except docker.errors.ImageNotFound:
            raise RuntimeError("Docker image not found.")
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while running the container: {e}")
        finally:
            cleanup_container(container)


class QTDockerEnv(DockerEnv):
    """Qlib Torch Docker"""

    def __init__(self, conf: DockerConf = QlibDockerConf()):
        super().__init__(conf)

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Download image & data if it doesn't exist
        """
        super().prepare()
        qlib_data_path = next(iter(self.conf.extra_volumes.keys()))
        if not (Path(qlib_data_path) / "qlib_data" / "cn_data").exists():
            logger.info("We are downloading!")
            cmd = "python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --interval 1d --delete_old False"
            self.check_output(entry=cmd)
        else:
            logger.info("Data already exists. Download skipped.")


class KGDockerEnv(DockerEnv):
    """Kaggle Competition Docker"""

    def __init__(self, competition: str | None = None, conf: DockerConf = KGDockerConf()):
        super().__init__(conf)


class MLEBDockerEnv(DockerEnv):
    """MLEBench Docker"""

    def __init__(self, conf: DockerConf = MLEBDockerConf()):
        super().__init__(conf)
