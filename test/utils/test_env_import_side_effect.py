import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]


def _run_in_subprocess(script: str, trace_path: Path) -> None:
    environment = os.environ.copy()
    environment["LOG_TRACE_PATH"] = str(trace_path)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=environment, check=True)


@pytest.mark.offline
def test_importing_env_module_does_not_create_log_directory(tmp_path: Path) -> None:
    """Importing ``rdagent.utils.env`` must not touch the file system.

    The docker env classes used to instantiate their configuration in the default
    argument of ``__init__``, which is evaluated at class definition time.  That
    triggered pydantic validation and the declarative ``share.yaml`` lookup, which
    renders a jinja template whose logging created a directory for the current
    trace path -- so merely importing the module left an empty log directory
    behind (``<trace_path>/debug_tpl/<pid>/``).

    A subprocess is required: in-process the module is already imported by other
    import chains, so the import-time side effect cannot be observed there.
    """
    trace_path = tmp_path / "trace"
    _run_in_subprocess(
        """
import os
from pathlib import Path

import rdagent.utils.env as env

assert env.FTDockerConf is not None

trace_path = Path(os.environ["LOG_TRACE_PATH"])
assert not trace_path.exists(), f"importing rdagent.utils.env created {trace_path}"
""",
        trace_path,
    )
    assert not trace_path.exists()


@pytest.mark.offline
def test_docker_env_subclasses_default_to_their_own_conf(tmp_path: Path) -> None:
    """Deferring the default configuration must not change the defaults."""
    _run_in_subprocess(
        """
from rdagent.utils.env import (
    BenchmarkDockerConf,
    BenchmarkDockerEnv,
    FTDockerConf,
    FTDockerEnv,
    KGDockerConf,
    KGDockerEnv,
    MLEBDockerConf,
    MLEBDockerEnv,
    QlibDockerConf,
    QTDockerEnv,
)

assert isinstance(QTDockerEnv().conf, QlibDockerConf)
assert isinstance(KGDockerEnv().conf, KGDockerConf)
assert isinstance(MLEBDockerEnv().conf, MLEBDockerConf)
assert isinstance(FTDockerEnv().conf, FTDockerConf)
assert isinstance(BenchmarkDockerEnv().conf, BenchmarkDockerConf)

# an explicitly passed conf still takes precedence
custom = FTDockerConf(terminal_tail_lines=7)
assert FTDockerEnv(custom).conf is custom
""",
        tmp_path / "trace",
    )
