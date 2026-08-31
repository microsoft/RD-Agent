from types import SimpleNamespace

import pytest

import rdagent.components.coder.data_science.share.eval as share_eval
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator


class FakeTemplate:
    def r(self, *args, **kwargs):
        return "/input"


@pytest.mark.parametrize(
    ("change_scores", "change_submission", "expected_messages"),
    [
        (True, False, ("scores.csv",)),
        (False, True, ("submission.csv",)),
        (True, True, ("scores.csv", "submission.csv")),
        (False, False, ()),
    ],
)
def test_model_dump_high_check_reports_each_changed_artifact(
    tmp_path,
    monkeypatch,
    change_scores,
    change_submission,
    expected_messages,
):
    monkeypatch.setattr(share_eval, "T", lambda *args, **kwargs: FakeTemplate())
    monkeypatch.setattr(share_eval, "get_ds_env", lambda *args, **kwargs: object())
    monkeypatch.setattr(share_eval, "get_clear_ws_cmd", lambda *args, **kwargs: "clear")
    monkeypatch.setattr(share_eval, "remove_eda_part", lambda stdout: stdout)
    monkeypatch.setattr(share_eval.DS_RD_SETTING, "model_dump_check_level", "high")
    monkeypatch.setattr(
        share_eval,
        "build_cls_from_json_with_retry",
        lambda *args, **kwargs: CoSTEERSingleFeedback(
            execution="ok",
            return_checking="existing diagnostics",
            code="ok",
            final_decision=True,
        ),
    )

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.bin").write_text("model", encoding="utf-8")

    scores_before = "model,score\nensemble,0.1\n"
    scores_after = "model,score\nensemble,0.2\n"
    submission_before = "id,pred\n1,0.1\n"
    submission_after = "id,pred\n1,0.2\n"
    (tmp_path / "scores.csv").write_text(scores_before, encoding="utf-8")
    (tmp_path / "submission.csv").write_text(submission_before, encoding="utf-8")

    class FakeImplementation:
        workspace_path = tmp_path
        all_codes = {"main.py": "pass"}

        def execute(self, env=None, entry=None):
            if entry == "clear":
                (tmp_path / "scores.csv").unlink(missing_ok=True)
                (tmp_path / "submission.csv").unlink(missing_ok=True)
                return "cleared"

            assert "--inference" in entry
            (tmp_path / "scores.csv").write_text(
                scores_after if change_scores else scores_before,
                encoding="utf-8",
            )
            (tmp_path / "submission.csv").write_text(
                submission_after if change_submission else submission_before,
                encoding="utf-8",
            )
            return "inference finished"

    scen = SimpleNamespace(
        competition="demo-competition",
        debug_path="/tmp/demo-input",
        real_debug_timeout=lambda: 1,
        real_full_timeout=lambda: 1,
    )
    feedback = ModelDumpEvaluator(scen, data_type="sample").evaluate(
        None,
        FakeImplementation(),
        None,
    )

    diagnostics = feedback.return_checking or ""
    assert "existing diagnostics" in diagnostics
    for filename in expected_messages:
        assert f"content of {filename} has changed" in diagnostics
    for filename in {"scores.csv", "submission.csv"} - set(expected_messages):
        assert f"content of {filename} has changed" not in diagnostics
