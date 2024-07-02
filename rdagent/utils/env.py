"""
The motiviation of the utils is for environment management

Tries to create uniform environment for the agent to run;
- All the code and data is expected included in one folder

"""
import docker
from abc import abstractmethod
from pydantic import BaseModel
from typing import Generic, TypeVar

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
    def run(self, local_path: str, entry: str | None, env: dict | None = None):
        """
        Run the folder under the environment.

        Parameters
        ----------
        local_path : str
            the local path (to project, mainly for code) will be mounted into the docker
        entry : str | None
            We may we the entry point when we run it.
            For example, we may have different entries when we run and summarize the project.
        env : dict | None
            Run the code with your specific environment.
        """


## Local Environment -----


class LocalConf(BaseModel):
    py_entry: str  # where you can find your python path


class LocalEnv(Env[LocalConf]):
    """
    Sometimes local environment may be more convinient for testing
    """
    conf: LocalConf


## Docker Environment -----


class DockerConf(BaseModel):
    image: str  # the image you want to run
    mount_path: str  # the path in the docker image to mount the folder
    default_entry: str  # the entry point of the image


QLIB_TORCH_IMAGE = DockerConf(image="linlanglv/qlib_image_nightly_pytorch:nightly",
                              mount_path="/workspace",
                              default_entry="qrun conf.yaml")


class DockerEnv(Env[DockerConf]):

    def prepare(self):
        """
        Download image if it doesn't exist
        """
        # TODO: download the image
        client = docker.from_env()
        try:
            client.images.get(self.conf.image)
        except docker.errors.ImageNotFound:
            client.images.pull(self.conf.image)
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while pulling the image: {e}")

    def run(self, local_path: str, entry: str | None, env: dict | None = None):

        if env is None:
            env = {}
        # TODO:
        # - Mount the local_path to mount_path
        # - run with entry
        client = docker.from_env()
        if entry is None:
            entry = self.conf.default_entry

        try:
            container = client.containers.run(
                image=self.conf.image,
                command=entry,
                volumes={local_path: {'bind': self.conf.mount_path, 'mode': 'rw'}},
                environment=env,
                detach=True
            )
            logs = container.logs(stream=True)
            for log in logs:
                print(log.strip().decode())
            container.wait()
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
