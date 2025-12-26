
from pathlib import Path
from pydantic_settings.main import SettingsConfigDict
from rdagent.utils.env import DockerConf, DockerEnv
from rdagent.app.data_science.conf import DS_RD_SETTING
import logging
import shutil

logger = logging.getLogger(__name__)


class AgentSysDockerConf(DockerConf):
    # TODO: change the content
    model_config = SettingsConfigDict(env_prefix="ASYS_DOCKER_")

    build_from_dockerfile: bool = True

    dockerfile_folder_path: Path = Path(__file__).parent / "docker"
    image: str = "local_agentic_sys:latest"


    #Mount and execution strategy
    mount_path: str = "/workspace/rdagent-solution"
    #mount_path: str = "/workspace"


    default_entry: str = "python main.py"
    #default_entry: str = "python train.py"

    running_timeout_period: int | None = 600
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )

def sanitize_container_path(path):
    p = path.replace("\\","/")
    if ":" in p:
        #remove drive letter
        p = p.split(":",1)[-1]
    if not p.startswith("/"):
        p = "/" + p.lstrip("/")
    return p

def build_volume(ws_path, mount_path, extra):
    """
    return Docker SDK volume mapping dict
    """
    vols = {}
    host_ws = str(ws_path.resolve())
    container_ws = sanitize_container_path(mount_path)
    vols[host_ws] = {"bind": container_ws, "mode": "rw"}
    if extra:
        for host, container in extra.items():
            host_res = str(Path(host).resolve())
            container_res = sanitize_container_path(container)
            vols[host_res] = {"bind": container_res, "mode": "rw"}
    return vols




def get_agent_sys_env(
    extra_volumes: dict = {},
    running_timeout_period: int | None = DS_RD_SETTING.debug_timeout,
    enable_cache: bool | None = None,
) -> DockerEnv:
    """
    create and prepare Docker environment for agentic system scenario
    """
    conf = AgentSysDockerConf()
    env = DockerEnv(conf=conf)
    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


# def get_agent_sys_env(
#     extra_volumes:dict = {},
#     running_timeout_period: int | None = DS_RD_SETTING.debug_timeout,
#     enable_cache: bool | None = None,
# ) -> DockerEnv:
#     """
#     create and prepare Docker environment for agentic system scenario
#     """
#     conf = AgentSysDockerConf()
#     env = DockerEnv(conf=conf)
#     env.conf.extra_volumes = extra_volumes.copy()
#     env.conf.running_timeout_period = running_timeout_period
#     if enable_cache is not None:
#         env.conf.enable_cache = enable_cache
#     #inject correct volumes before preparation
#     env.conf.mount_path = sanitize_container_path(env.conf.mount_path)
    
#     # 清理 extra_volumes 中的容器路径
#     if env.conf.extra_volumes:
#         sanitized_extra = {}
#         for host, container in env.conf.extra_volumes.items():
#             sanitized_extra[host] = sanitize_container_path(container)
#         env.conf.extra_volumes = sanitized_extra
    
#     env.prepare()
#     return env


    

