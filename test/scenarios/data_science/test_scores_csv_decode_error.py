from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import rdagent.components.coder.data_science.pipeline.eval as pipeline_eval
import rdagent.components.coder.data_science.workflow.eval as workflow_eval
import rdagent.scenarios.data_science.dev.runner.eval as runner_eval
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import CoSTEEREvaluator, CoSTEERSingleFeedback


@pytest.mark.offline
@pytest.mark.parametrize(
    ("module", "evaluator_type", "feedback_type"),
    [
        (workflow_eval, workflow_eval.WorkflowGeneralCaseSpecEvaluator, workflow_eval.WorkflowSingleFeedback),
        (pipeline_eval, pipeline_eval.PipelineCoSTEEREvaluator, pipeline_eval.PipelineSingleFeedback),
        (runner_eval, runner_eval.DSRunnerEvaluator, runner_eval.DSRunnerFeedback),
    ],
    ids=["workflow", "pipeline", "runner"],
)
@pytest.mark.parametrize(
    ("content", "valid"),
    [
        (b"\xff\xff\xff\n", False),
        (b"model,rmse\nensemble,\xc3\n", False),
        (b"", False),
        ('model,rmse\n"café,0.25\n'.encode(), False),
        (b"model,rmse\nensemble,0.25\n", True),
    ],
    ids=["invalid-utf8", "truncated-utf8", "empty", "malformed-unicode-csv", "valid"],
)
def test_score_diagnostics_preserve_feedback(
    module: ModuleType,
    evaluator_type: type[CoSTEEREvaluator],
    feedback_type: type[CoSTEERSingleFeedback],
    content: bytes,
    valid: bool,  # noqa: FBT001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_path = tmp_path / "scores.csv"
    score_path.write_bytes(content)
    env = SimpleNamespace(conf=SimpleNamespace(running_timeout_period=10))
    monkeypatch.setattr(module, "get_ds_env", Mock(return_value=env))
    monkeypatch.setattr(module, "get_clear_ws_cmd", Mock(return_value="clear"))
    monkeypatch.setattr(module, "T", Mock(return_value=SimpleNamespace(r=Mock(return_value="prompt"))))
    for setting in (
        "sample_data_by_LLM",
        "enable_notebook_conversion",
        "enable_mcp_documentation_search",
        "only_first_loop_enable_hyperparameter_tuning",
        "only_enable_tuning_in_merge",
    ):
        monkeypatch.setattr(DS_RD_SETTING, setting, False)
    monkeypatch.setattr(DS_RD_SETTING, "coder_on_whole_pipeline", True)
    if hasattr(module, "get_test_eval"):
        test_evaluator = SimpleNamespace(
            enabled=Mock(return_value=False),
            is_sub_enabled=Mock(return_value=False),
            get_sample_submission_name=Mock(return_value=None),
        )
        monkeypatch.setattr(module, "get_test_eval", Mock(return_value=test_evaluator))
    monkeypatch.setattr(
        runner_eval,
        "RD_Agent_TIMER_wrapper",
        SimpleNamespace(timer=SimpleNamespace(remain_time=Mock(return_value=None), all_duration=None)),
    )
    feedback = feedback_type(execution="ok", return_checking="original feedback", code="ok", final_decision=True)
    if isinstance(feedback, runner_eval.DSRunnerFeedback):
        feedback.acceptable = True
    build_feedback = Mock(return_value=feedback)
    monkeypatch.setattr(module, "build_cls_from_json_with_retry", build_feedback)
    scenario = SimpleNamespace(
        debug_path="/unused",
        competition="test",
        metric_name="rmse",
        real_debug_timeout=Mock(return_value=10),
        real_full_timeout=Mock(return_value=10),
        get_scenario_all_desc=Mock(return_value="scenario"),
    )
    workspace = SimpleNamespace(
        workspace_path=tmp_path,
        file_dict={"main.py": "pass", "spec/workflow.md": "spec"},
        all_codes="pass",
        change_summary="unchanged",
        running_info=SimpleNamespace(running_time=0),
        execute=Mock(return_value="ok"),
        run=Mock(return_value=SimpleNamespace(stdout="ok", exit_code=0, running_time=1)),
        inject_files=Mock(),
    )
    knowledge = SimpleNamespace(
        success_task_to_knowledge_dict={},
        failed_task_info_set=set(),
        task_to_similar_task_successful_knowledge={"task": []},
        task_to_former_failed_traces={"task": ([], None)},
    )
    task = SimpleNamespace(get_task_information=Mock(return_value="task"))
    result = evaluator_type(scen=scenario).evaluate(task, workspace, None, queried_knowledge=knowledge)

    assert result is feedback
    assert result.final_decision is valid
    assert result.return_checking is not None
    assert result.return_checking.startswith("original feedback")
    assert score_path.read_bytes() == content
    build_feedback.assert_called_once()
    if not valid:
        assert "[Error] in checking the scores.csv file:" in result.return_checking
        assert content.decode("utf-8", errors="replace") in result.return_checking
    else:
        assert result.return_checking == "original feedback"
