from pathlib import Path

from rdagent.core.experiment import FBWorkspace
from rdagent.utils.env import Env


def get_runtime_environment_by_env(env: Env) -> str:
    implementation = FBWorkspace()
    fname = "runtime_info.py"
    implementation.inject_files(**{fname: (Path(__file__).absolute().resolve().parent / "runtime_info.py").read_text()})
    stdout = implementation.execute(env=env, entry=f"python {fname}")
    return stdout
