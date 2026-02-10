"""
RL Training Environment Configuration
"""
from pathlib import Path

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.rl.autorl_bench.utils.download import download_data, download_model
from rdagent.utils.env import DockerConf, DockerEnv

# RL 资源路径（统一使用 RL_RD_SETTING.file_path）
RL_MODELS_DIR = RL_RD_SETTING.file_path / "models"
RL_DATA_DIR = RL_RD_SETTING.file_path / "datasets"
RL_WORKSPACE_DIR = Path(__file__).parent.parent / "eval" / "autorl_bench" / "example_workspace"

# Dockerfile 所在目录
RL_DOCKERFILE_DIR = Path(__file__).parent / "docker"


class RLDockerConf(DockerConf):
    """RL Docker 配置"""
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = RL_DOCKERFILE_DIR
    dockerfile: str = "base/Dockerfile"
    image: str = "autorl-bench/base:latest"
    mount_path: str = "/workspace"
    default_entry: str = "python main.py"
    running_timeout_period: int = 3600
    enable_cache: bool = False
    save_logs_to_file: bool = True


def get_rl_env(benchmark: str = "base", timeout: int = 3600) -> DockerEnv:
    """
    获取 RL 训练环境
    
    根据 benchmark 自动选择对应的 Docker 镜像：
    - benchmark="gsm8k" → autorl-bench/gsm8k:latest
    - benchmark="evalplus" → autorl-bench/evalplus:latest
    - benchmark="base" → autorl-bench/base:latest
    
    如果镜像已存在，跳过构建；否则自动构建。
    如果数据/模型不存在，自动下载。
    
    Args:
        benchmark: benchmark 名称，决定使用哪个 Docker 镜像
        timeout: 运行超时时间（秒）
        
    Returns:
        配置好的 DockerEnv
    """
    # 下载数据（download_data 内部有存在检查）
    if benchmark != "base":
        download_data(benchmark, str(RL_DATA_DIR))
    
    # 下载模型（download_model 内部有存在检查）
    base_model = RL_RD_SETTING.base_model
    if base_model:
        download_model(base_model, str(RL_MODELS_DIR))
    
    conf = RLDockerConf()
    conf.running_timeout_period = timeout
    
    # 根据 benchmark 设置镜像名和 Dockerfile 目录
    conf.image = f"autorl-bench/{benchmark}:latest"
    conf.dockerfile_folder_path = RL_DOCKERFILE_DIR / benchmark
    # 基类 DockerEnv.prepare() 会在 dockerfile_folder_path 下找 Dockerfile
    
    # 挂载目录
    conf.extra_volumes = {
        str(RL_MODELS_DIR): {"bind": "/models", "mode": "ro"},
        str(RL_DATA_DIR): {"bind": "/data", "mode": "ro"},
    }
    
    env = DockerEnv(conf=conf)
    env.prepare()
    
    logger.info(f"RL DockerEnv prepared: {conf.image}")
    logger.info(f"  Models: {RL_MODELS_DIR} -> /models")
    logger.info(f"  Data: {RL_DATA_DIR} -> /data")
    
    return env


# RL Benchmark 资源路径
RL_BENCHMARKS_DIR = RL_RD_SETTING.file_path / "benchmarks"


class RLBenchmarkDockerConf(DockerConf):
    """RL Benchmark Docker 配置（OpenCompass）"""
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent.parent.parent / "components" / "coder" / "finetune" / "env" / "docker" / "opencompass"
    image: str = "rdagent-opencompass:latest"
    mount_path: str = "/workspace"
    default_entry: str = "opencompass --help"
    running_timeout_period: int = 3600
    enable_cache: bool = False
    save_logs_to_file: bool = True
    network: str | None = "host"
    env_dict: dict = {"COMPASS_DATA_CACHE": "/benchmarks/opencompass_data"}


def get_rl_benchmark_env(timeout: int = 3600) -> DockerEnv:
    """
    获取 RL 评测环境（OpenCompass）
    
    独立的 RL benchmark 环境，不依赖 SFT。
    
    Args:
        timeout: 运行超时时间（秒）
        
    Returns:
        配置好的 DockerEnv
    """
    # 确保 benchmarks 目录存在
    RL_BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    
    conf = RLBenchmarkDockerConf()
    conf.running_timeout_period = timeout
    
    # 挂载 RL 的 benchmarks 目录（使用绝对路径绑定）
    conf.extra_volumes = {
        str(RL_BENCHMARKS_DIR.resolve()): {"bind": "/benchmarks", "mode": "rw"},
        str(RL_MODELS_DIR.resolve()): {"bind": "/models", "mode": "ro"},
    }
    
    env = DockerEnv(conf=conf)
    env.prepare()
    
    logger.info(f"RL Benchmark DockerEnv prepared")
    logger.info(f"  Benchmarks: {RL_BENCHMARKS_DIR} -> /benchmarks")
    logger.info(f"  Models: {RL_MODELS_DIR} -> /models")
    
    return env
