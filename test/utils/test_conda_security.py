from pathlib import Path
from unittest.mock import Mock

import pytest

from rdagent.utils.env import _prepare_conda_env


@pytest.mark.offline
def test_prepare_conda_env_rejects_shell_metacharacters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    subprocess_run = Mock()
    monkeypatch.setattr("rdagent.utils.env.subprocess.run", subprocess_run)
    requirements = tmp_path / "requirements.txt"
    requirements.touch()

    with pytest.raises(ValueError, match="Invalid conda environment name"):
        _prepare_conda_env("safe; touch /tmp/pwned", requirements)
    subprocess_run.assert_not_called()


@pytest.mark.offline
def test_prepare_conda_env_uses_argument_lists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    check_call = Mock()
    monkeypatch.setattr("rdagent.utils.env._sync_conda_cache_with_real_envs", Mock())
    monkeypatch.setattr("rdagent.utils.env.subprocess.check_call", check_call)
    monkeypatch.setattr("rdagent.utils.env._CONDA_ENV_PREPARED", set())
    requirements = tmp_path / "requirements.txt"
    requirements.touch()

    _prepare_conda_env("safe-env", requirements)

    assert check_call.call_args_list[0].args[0] == ["conda", "create", "-y", "-n", "safe-env", "python=3.10"]
    assert all(isinstance(call.args[0], list) for call in check_call.call_args_list)
