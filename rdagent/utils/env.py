"""
The motiviation of the utils is for environment management

Tries to create uniform environment for the agent to run;
- All the code and data is expected included in one folder

"""
import os
import sys
import docker
import subprocess
from abc import abstractmethod
from pydantic import BaseModel
from typing import Generic, TypeVar, Optional, Dict
from pathlib import Path

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

    def run(self,
            entry: str | None = None,
            local_path: Optional[str] = None,
            env: dict | None = None) -> str:
        if env is None:
            env = {}

        if entry is None:
            entry = self.conf.default_entry

        command = str(Path(self.conf.py_bin).joinpath(entry)).split(" ")

        cwd = None
        if local_path:
            cwd = Path(local_path).resolve()
        print(command)
        result = subprocess.run(
            command,
            cwd=cwd,
            env={**os.environ, **env},
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Error while running the command: {result.stderr}")

        return result.stdout


## Docker Environment -----


class DockerConf(BaseModel):
    image: str  # the image you want to run
    mount_path: str  # the path in the docker image to mount the folder
    default_entry: str  # the entry point of the image

    extra_volumes: dict | None = {}
    # Sometime, we need maintain some extra data for the workspace.
    # And the extra data may be shared and the downloading can be time consuming.
    # So we just want to download it once.


QLIB_TORCH_IMAGE = DockerConf(
    image="linlanglv/qlib_image_nightly_pytorch:nightly",
    mount_path="/workspace",
    default_entry="qrun conf.yaml",
    extra_volumes={Path("~/.qlib/").expanduser().resolve(): "/root/.qlib/"},
)


class DockerEnv(Env[DockerConf]):
    # TODO: Save the output into a specific file

    def prepare(self):
        """
        Download image if it doesn't exist
        """
        client = docker.from_env()
        try:
            client.images.get(self.conf.image)
        except docker.errors.ImageNotFound:
            client.images.pull(self.conf.image)
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while pulling the image: {e}")

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
            container = client.containers.run(
                image=self.conf.image,
                command=entry,
                volumes=volumns,
                environment=env,
                detach=True,
                working_dir=self.conf.mount_path,
                auto_remove=True,
            )
            logs = container.logs(stream=True)
            for log in logs:
                decoded_log = log.strip().decode()
                print(decoded_log)
                log_output += decoded_log + "\n"
            container.wait()
            return log_output
        except docker.errors.ContainerError as e:
            raise RuntimeError(f"Error while running the container: {e}")
        except docker.errors.ImageNotFound:
            raise RuntimeError("Docker image not found.")
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while running the container: {e}")


class QTDockerEnv(DockerEnv):
    """Qlib Torch Docker"""

    def __init__(self, conf: DockerConf = QLIB_TORCH_IMAGE):
        super().__init__(conf)

    def prepare(self):
        """
        Download image & data if it doesn't exist
        """
        super().prepare()
        qlib_data_path = next(iter(self.conf.extra_volumes.keys()))
        if not (Path(qlib_data_path) / "qlib_data" / "cn_data").exists():
            cmd = "python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --interval 1d --delete_old False"
            self.run(entry=cmd)
        else:
            print("Data already exists. Download skipped.")
