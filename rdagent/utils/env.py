"""
The motiviation of the utils is for environment management

Tries to create uniform environment for the agent to run;
- All the code and data is expected included in one folder
"""
# TODO: move the scenario specific docker env into other folders.

import os
import subprocess
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Generic, Optional, TypeVar

import docker
import docker.models
import docker.models.containers
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from rdagent.log import rdagent_logger as logger

ASpecificBaseModel = TypeVar("ASpecificBaseModel", bound=BaseModel)


class Env(Generic[ASpecificBaseModel]):
    """
    We use BaseModel as the setting due to the featurs it provides
    - It provides base typing and checking featurs.
    - loading and dumping the information will be easier: for example, we can use package like `pydantic-yaml`
    """

    conf: ASpecificBaseModel  # different env have different conf.

    def __init__(self, conf: ASpecificBaseModel):
        self.conf = conf

    @abstractmethod
    def prepare(self):
        """
        Prepare for the environment based on it's configure
        """

    @abstractmethod
    def run(self, entry: str | None, local_path: str | None = None, env: dict | None = None) -> str:
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

    def prepare(self):
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


class DockerConf(BaseSettings):
    build_from_dockerfile: bool = False
    dockerfile_folder_path: Optional[
        Path
    ] = None  # the path to the dockerfile optional path provided when build_from_dockerfile is False
    image: str  # the image you want to build
    mount_path: str  # the path in the docker image to mount the folder
    default_entry: str  # the entry point of the image

    extra_volumes: dict | None = {}
    # Sometime, we need maintain some extra data for the workspace.
    # And the extra data may be shared and the downloading can be time consuming.
    # So we just want to download it once.
    network: str | None = "bridge"  # the network mode for the docker
    shm_size: str | None = None
    enable_gpu: bool = True  # because we will automatically disable GPU if not available. So we enable it by default.


class QlibDockerConf(DockerConf):
    class Config:
        env_prefix = "QLIB_DOCKER_"  # Use QLIB_DOCKER_ as prefix for environment variables

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "qlib" / "docker"
    image: str = "local_qlib:latest"
    mount_path: str = "/workspace/qlib_workspace/"
    default_entry: str = "qrun conf.yaml"
    extra_volumes: dict = {Path("~/.qlib/").expanduser().resolve(): "/root/.qlib/"}
    shm_size: str | None = "16g"
    enable_gpu: bool = True


class DMDockerConf(DockerConf):
    class Config:
        env_prefix = "DM_DOCKER_"

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "data_mining" / "docker"
    image: str = "local_dm:latest"
    mount_path: str = "/workspace/dm_workspace/"
    default_entry: str = "python train.py"
    extra_volumes: dict = {
        Path("~/.rdagent/.data/physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3/")
        .expanduser()
        .resolve(): "/root/.data/"
    }
    shm_size: str | None = "16g"


# physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3
class DockerEnv(Env[DockerConf]):
    # TODO: Save the output into a specific file

    def prepare(self):
        """
        Download image if it doesn't exist
        """
        client = docker.from_env()
        if self.conf.build_from_dockerfile and self.conf.dockerfile_folder_path.exists():
            logger.info(f"Building the image from dockerfile: {self.conf.dockerfile_folder_path}")
            image, logs = client.images.build(
                path=str(self.conf.dockerfile_folder_path), tag=self.conf.image, network_mode=self.conf.network
            )
            logger.info(f"Finished building the image from dockerfile: {self.conf.dockerfile_folder_path}")
        try:
            client.images.get(self.conf.image)
        except docker.errors.ImageNotFound:
            client.images.pull(self.conf.image)
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while pulling the image: {e}")

    def _gpu_kwargs(self, client):
        """get gpu kwargs based on its availability"""
        if not self.conf.enable_gpu:
            return {}
        gpu_kwargs = {
            "device_requests": [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
            if self.conf.enable_gpu
            else None,
        }
        try:
            client.containers.run(self.conf.image, "nvidia-smi", **gpu_kwargs)
            logger.info("GPU Devices are available.")
        except docker.errors.APIError:
            return {}
        return gpu_kwargs

    def run(self, entry: str | None = None, local_path: str | None = None, env: dict | None = None):
        if env is None:
            env = {}
        client = docker.from_env()
        if entry is None:
            entry = self.conf.default_entry

        volumns = {}
        if local_path is not None:
            local_path = os.path.abspath(local_path)
            volumns[local_path] = {"bind": self.conf.mount_path, "mode": "rw"}
        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumns[lp] = {"bind": rp, "mode": "rw"}

        log_output = ""

        try:
            container: docker.models.containers.Container = client.containers.run(
                image=self.conf.image,
                command=entry,
                volumes=volumns,
                environment=env,
                detach=True,
                working_dir=self.conf.mount_path,
                # auto_remove=True, # remove too fast might cause the logs not to be get
                network=self.conf.network,
                shm_size=self.conf.shm_size,
                **self._gpu_kwargs(client),
            )
            logs = container.logs(stream=True)
            for log in logs:
                decoded_log = log.strip().decode()
                print(decoded_log)
                log_output += decoded_log + "\n"
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


class QTDockerEnv(DockerEnv):
    """Qlib Torch Docker"""

    def __init__(self, conf: DockerConf = QlibDockerConf()):
        super().__init__(conf)

    def prepare(self):
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

    def prepare(self, username: str, password: str):
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
