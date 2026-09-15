from pathlib import Path
from unittest.mock import MagicMock, patch

from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "rdagent" / "scenarios" / "qlib" / "experiment" / "factor_template"


def test_execute_missing_result_includes_read_exp_res_log():
    """When qrun succeeds but read_exp_res.py fails to produce ret.parquet (e.g. because the
    backtest artifact it depends on was never written), the returned log should still contain
    read_exp_res.py's own output so the failure reason is not silently dropped."""
    workspace = QlibFBWorkspace(template_folder_path=TEMPLATE_PATH)

    mock_env = MagicMock()
    mock_env.check_output.side_effect = [
        "qrun finished without error",
        "Traceback (most recent call last):\nLoadObjectError: No such file or directory",
    ]

    with patch("rdagent.scenarios.qlib.experiment.workspace.QlibCondaEnv", return_value=mock_env), patch(
        "rdagent.components.coder.model_coder.conf.MODEL_COSTEER_SETTINGS.env_type", "conda"
    ):
        result, log = workspace.execute()

    assert result is None
    assert "qrun finished without error" in log
    assert "LoadObjectError" in log
