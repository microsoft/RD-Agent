from pathlib import Path
from types import SimpleNamespace

import pytest

import rdagent.components.coder.data_science.share.eval as share_eval
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator


def test_model_dump_evaluator_rejects_empty_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(share_eval, "get_ds_env", lambda **_: None)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.bin").touch()
    (tmp_path / "submission.csv").write_text("id,pred\n1,0.5\n", encoding="utf-8")
    (tmp_path / "scores.csv").touch()
    implementation = SimpleNamespace(
        workspace_path=tmp_path,
        all_codes={"main.py": "pass"},
        execute=lambda **_: "inference finished",
    )
    scenario = SimpleNamespace(debug_path=tmp_path, real_debug_timeout=lambda: 1)

    feedback = ModelDumpEvaluator(scenario, data_type="sample").evaluate(None, implementation, None)

    assert feedback.final_decision is False
    assert feedback.return_checking is not None
    assert "scores.csv" in feedback.return_checking
    assert "empty" in feedback.return_checking
