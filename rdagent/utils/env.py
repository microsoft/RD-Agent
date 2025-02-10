"""
The motiviation of the utils is for environment management

Tries to create uniform environment for the agent to run;
- All the code and data is expected included in one folder
"""

# TODO: move the scenario specific docker env into other folders.

import json
import os
import pickle
import re
import subprocess
import time
import uuid
import zipfile
from abc import abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

import docker  # type: ignore[import-untyped]
import docker.models  # type: ignore[import-untyped]
import docker.models.containers  # type: ignore[import-untyped]
import docker.types  # type: ignore[import-untyped]
from pydantic import BaseModel
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table

from rdagent.core.conf import ExtendedBaseSettings, ExtendedSettingsConfigDict
from rdagent.core.experiment import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.workflow import wait_retry

ASpecificBaseModel = TypeVar("ASpecificBaseModel", bound=BaseModel)


class Env(Generic[ASpecificBaseModel]):
    """
    We use BaseModel as the setting due to the features it provides
    - It provides base typing and checking features.
    - loading and dumping the information will be easier: for example, we can use package like `pydantic-yaml`
    """

    conf: ASpecificBaseModel  # different env have different conf.
    # last_exit_code:  # TODO: get the more concrete information about the exit code.

    def __init__(self, conf: ASpecificBaseModel):
        self.conf = conf

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Prepare for the environment based on it's configure
        """

    @abstractmethod
    def run(self, entry: str | None, local_path: str = ".", env: dict | None = None) -> str:
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


## Local Environment -----


class LocalConf(BaseModel):
    py_bin: str
    default_entry: str


class LocalEnv(Env[LocalConf]):
    """
    Sometimes local environment may be more convinient for testing
    """

    def prepare(self) -> None:
        if not (Path("~/.qlib/qlib_data/cn_data").expanduser().resolve().exists()):
            self.run(
                entry="python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn",
            )
        else:
            print("Data already exists. Download skipped.")

    def run(self, entry: str | None = None, local_path: Optional[str] = None, env: dict | None = None) -> str:
        if env is None:
            env = {}

        if entry is None:
            entry = self.conf.default_entry

        command = str(Path(self.conf.py_bin).joinpath(entry)).split(" ")

        cwd = None
        if local_path:
            cwd = Path(local_path).resolve()
        result = subprocess.run(command, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Error while running the command: {result.stderr}")

        return result.stdout


## Docker Environment -----


class DockerConf(ExtendedBaseSettings):
    build_from_dockerfile: bool = False
    dockerfile_folder_path: Optional[Path] = (
        None  # the path to the dockerfile optional path provided when build_from_dockerfile is False
    )
    image: str  # the image you want to build
    mount_path: str  # the path in the docker image to mount the folder
    default_entry: str  # the entry point of the image

    extra_volumes: dict = {}
    extra_volume_mode: str = "ro"  # by default. only the mount_path should be writable, others are changed to read-only
    # Sometime, we need maintain some extra data for the workspace.
    # And the extra data may be shared and the downloading can be time consuming.
    # So we just want to download it once.
    network: str | None = "bridge"  # the network mode for the docker
    shm_size: str | None = None
    enable_gpu: bool = True  # because we will automatically disable GPU if not available. So we enable it by default.
    mem_limit: str | None = "48g"  # Add memory limit attribute

    running_timeout_period: int = 3600  # 1 hour

    enable_cache: bool = True  # enable the cache mechanism


class QlibDockerConf(DockerConf):
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "qlib" / "docker"
    image: str = "local_qlib:latest"
    mount_path: str = "/workspace/qlib_workspace/"
    default_entry: str = "qrun conf.yaml"
    extra_volumes: dict = {str(Path("~/.qlib/").expanduser().resolve().absolute()): "/root/.qlib/"}
    shm_size: str | None = "16g"
    enable_gpu: bool = True


class DMDockerConf(DockerConf):
    model_config = ExtendedSettingsConfigDict(env_prefix="DM_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "data_mining" / "docker"
    image: str = "local_dm:latest"
    mount_path: str = "/workspace/dm_workspace/"
    default_entry: str = "python train.py"
    extra_volumes: dict = {
        str(
            Path("~/.rdagent/.data/physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3/")
            .expanduser()
            .resolve()
            .absolute()
        ): "/root/.data/"
    }
    shm_size: str | None = "16g"


class KGDockerConf(DockerConf):
    model_config = ExtendedSettingsConfigDict(env_prefix="KG_DOCKER_")

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

    running_timeout_period: int = 600
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )


class DSDockerConf(DockerConf):
    model_config = ExtendedSettingsConfigDict(env_prefix="DS_DOCKER_")

    build_from_dockerfile: bool = False
    image: str = "gcr.io/kaggle-gpu-images/python:latest"
    mount_path: str = "/kaggle/workspace"
    default_entry: str = "python main.py"

    running_timeout_period: int = 600
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )


class MLEBDockerConf(DockerConf):
    model_config = ExtendedSettingsConfigDict(env_prefix="MLEB_DOCKER_")

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

        @wait_retry(5, 10)
        def _f() -> dict:
            try:
                client.containers.run(self.conf.image, "nvidia-smi", **gpu_kwargs)
                logger.info("GPU Devices are available.")
            except docker.errors.APIError:
                return {}
            return gpu_kwargs

        return _f()

    def replace_time_info(self, input_string: str) -> str:
        """To remove any time related information from the logs since it will destroy the cache mechanism"""
        """We currently set this function as default, but it can be changed in the future"""
        datetime_pattern = r"\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?\b"
        output_string = re.sub(datetime_pattern, "[DATETIME]", input_string)
        return output_string

    def __run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: dict | None = None,
        remove_timestamp: bool = True,
    ) -> str:
        if env is None:
            env = {}
        env["PYTHONWARNINGS"] = "ignore"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        client = docker.from_env()

        volumns = {}
        if local_path is not None:
            local_path = os.path.abspath(local_path)
            volumns[local_path] = {"bind": self.conf.mount_path, "mode": "rw"}
        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumns[lp] = {"bind": rp, "mode": self.conf.extra_volume_mode}
        if running_extra_volume is not None:
            for lp, rp in running_extra_volume.items():
                volumns[lp] = {"bind": rp, "mode": self.conf.extra_volume_mode}

        log_output = ""

        try:
            container: docker.models.containers.Container = client.containers.run(  # type: ignore[no-any-unimported]
                image=self.conf.image,
                command=entry,
                volumes=volumns,
                environment=env,
                detach=True,
                working_dir=self.conf.mount_path,
                # auto_remove=True, # remove too fast might cause the logs not to be get
                network=self.conf.network,
                shm_size=self.conf.shm_size,
                mem_limit=self.conf.mem_limit,  # Set memory limit
                **self._gpu_kwargs(client),
            )
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
            table.add_row("Volumns", "\n".join(f"{k}:{v}" for k, v in volumns.items()))
            print(table)
            for log in logs:
                decoded_log = log.strip().decode()
                decoded_log = self.replace_time_info(decoded_log) if remove_timestamp else decoded_log
                Console().print(decoded_log, markup=False)
                log_output += decoded_log + "\n"
            print(Rule("[bold green]Docker Logs End[/bold green]", style="dark_orange"))
            container.wait()
            container.stop()
            container.remove()
            return log_output
        except docker.errors.ContainerError as e:
            raise RuntimeError(f"Error while running the container: {e}")
        except docker.errors.ImageNotFound:
            raise RuntimeError("Docker image not found.")
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while running the container: {e}")

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
        with zipfile.ZipFile(zip_file_path, "r") as z:
            z.extractall(folder_path)

    def cached_run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: dict | None = None,
        remove_timestamp: bool = True,
    ) -> str:
        """
        Run the folder under the environment.
        Will cache the output and the folder diff for next round of running.
        Use the python codes and the parameters(entry, running_extra_volume) as key to hash the input.
        """
        target_folder = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str) / f"utils.env.run"
        target_folder.mkdir(parents=True, exist_ok=True)

        # we must add the information of data (beyound code) into the key.
        # Otherwise, all commands operating on data will become invalue (e.g. rm -r submission.csv)
        # So we recursively walk in the folder and add the sorted relative filename list as part of the key.
        data_key = []
        for path in Path(local_path).rglob("*"):
            p = str(path.relative_to(Path(local_path)))
            if p.startswith("__pycache__"):
                continue
            data_key.append(p)
        data_key = sorted(data_key)

        key = md5_hash(
            json.dumps(
                [
                    [str(path.relative_to(Path(local_path))), path.read_text()]
                    for path in sorted(Path(local_path).rglob("*.py"))
                ]
            )
            + json.dumps({"entry": entry, "running_extra_volume": running_extra_volume})
            + json.dumps({"extra_volumes": self.conf.extra_volumes})
            + json.dumps(data_key)
        )
        if Path(target_folder / f"{key}.pkl").exists() and Path(target_folder / f"{key}.zip").exists():
            with open(target_folder / f"{key}.pkl", "rb") as f:
                ret: str = pickle.load(f)
            self.unzip_a_file_into_a_folder(str(target_folder / f"{key}.zip"), local_path)
        else:
            ret = self.__run(entry, local_path, env, running_extra_volume, remove_timestamp)
            with open(target_folder / f"{key}.pkl", "wb") as f:
                pickle.dump(ret, f)
            self.zip_a_folder_into_a_file(local_path, str(target_folder / f"{key}.zip"))
        return ret

    def run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: dict | None = None,
    ) -> str:
        if entry is None:
            entry = self.conf.default_entry
        entry_add_timeout = (
            f"/bin/sh -c 'timeout {self.conf.running_timeout_period} {entry}; chmod -R 777 {self.conf.mount_path}'"
        )

        start = time.time()
        if self.conf.enable_cache:
            out = self.cached_run(entry_add_timeout, local_path, env, running_extra_volume)
        else:
            out = self.__run(entry_add_timeout, local_path, env, running_extra_volume, remove_timestamp=False)
        end = time.time()

        if end - start + 1 >= self.conf.running_timeout_period:
            out += f"\n\nThe running time exceeds {self.conf.running_timeout_period} seconds, so the process is killed."

        return out

    def dump_python_code_run_and_get_results(
        self,
        code: str,
        dump_file_names: list[str],
        local_path: str,
        env: dict | None = None,
        running_extra_volume: dict | None = None,
        code_dump_file_py_name: Optional[str] = None,
    ) -> tuple[str, list]:
        """
        Dump the code into the local path and run the code.
        """
        random_file_name = f"{uuid.uuid4()}.py" if code_dump_file_py_name is None else f"{code_dump_file_py_name}.py"
        with open(os.path.join(local_path, random_file_name), "w") as f:
            f.write(code)
        entry = f"python {random_file_name}"
        log_output = self.run(entry, local_path, env, running_extra_volume=running_extra_volume)
        results = []
        os.remove(os.path.join(local_path, random_file_name))
        for name in dump_file_names:
            if os.path.exists(os.path.join(local_path, f"{name}")):
                results.append(pickle.load(open(os.path.join(local_path, f"{name}"), "rb")))
                os.remove(os.path.join(local_path, f"{name}"))
            else:
                return log_output, []
        return log_output, results


class QTDockerEnv(DockerEnv):
    """Qlib Torch Docker"""

    def __init__(self, conf: DockerConf = QlibDockerConf()):
        super().__init__(conf)

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[explicit-override, no-untyped-def]
        """
        Download image & data if it doesn't exist
        """
        super().prepare()
        qlib_data_path = next(iter(self.conf.extra_volumes.keys()))
        if not (Path(qlib_data_path) / "qlib_data" / "cn_data").exists():
            logger.info("We are downloading!")
            cmd = "python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --interval 1d --delete_old False"
            self.run(entry=cmd)
        else:
            logger.info("Data already exists. Download skipped.")


class DMDockerEnv(DockerEnv):
    """Qlib Torch Docker"""

    def __init__(self, conf: DockerConf = DMDockerConf()):
        super().__init__(conf)

    def prepare(self, username: str, password: str) -> None:
        """
        Download image & data if it doesn't exist
        """
        super().prepare()
        data_path = next(iter(self.conf.extra_volumes.keys()))
        if not (Path(data_path)).exists():
            logger.info("We are downloading!")
            cmd = "wget -r -N -c -np --user={} --password={} -P ~/.rdagent/.data/ https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/".format(
                username, password
            )
            os.system(cmd)
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
