import sys
import warnings
from pathlib import Path

from rdagent.core.experiment import FBWorkspace
from rdagent.utils.env import Env, LocalEnv


def get_runtime_environment_by_env(env: Env) -> str:
    implementation = FBWorkspace()
    fname = "runtime_info.py"
    implementation.inject_files(**{fname: (Path(__file__).absolute().resolve().parent / "runtime_info.py").read_text()})
    stdout = implementation.execute(env=env, entry=f"python {fname}")
    return stdout


def check_runtime_environment(env: Env) -> str:
    implementation = FBWorkspace()
    # 1) Check if strace exists in env
    strace_check = implementation.execute(env=env, entry="which strace || echo MISSING").strip()
    if strace_check.endswith("MISSING"):
        if isinstance(env, LocalEnv) and sys.platform == "darwin":
            # NOTE: since macos does not have strace, and dtruss need a root permission, we just ignore it
            warnings.warn("`strace` not support macOS, the final result may be different.", stacklevel=1)
        else:
            raise RuntimeError("`strace` not found in the target environment.")

    # 2) Check if coverage module works in env
    coverage_check = implementation.execute(env=env, entry="python -m coverage --version || echo MISSING").strip()
    if coverage_check.endswith("MISSING"):
        raise RuntimeError("`coverage` module not found or not runnable in the target environment.")
