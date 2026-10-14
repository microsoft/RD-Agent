import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from rdagent.scenarios.qlib.developer import feedback


class SyntheticTask:
    def get_task_information(self):
        return "Synthetic task."

    def get_task_information_and_implementation_result(self):
        return {
            "factor_name": "synthetic",
            "factor_description": "Synthetic task.",
            "factor_formulation": "x",
            "variables": {},
            "factor_implementation": "True",
        }


def results(metrics):
    return pd.DataFrame({"0": range(1, len(metrics) + 1)}, index=metrics, dtype=float)


def test_process_results_keeps_default_metrics_and_order():
    frame = results(feedback.IMPORTANT_METRICS + ["unrelated"])
    rendered = feedback.process_results(frame, frame)
    assert rendered.startswith("IC of Current Result is 1.000000")
    assert "unrelated" not in rendered


def test_process_results_accepts_explicit_metrics():
    frame = results(["precision_at_5", "valid_loss", "heldout_loss"])
    rendered = feedback.process_results(frame, frame, metrics=["valid_loss", "precision_at_5"])
    assert rendered.startswith("valid_loss of Current Result is 2.000000")
    assert "precision_at_5" in rendered
    assert "heldout_loss" not in rendered


@pytest.mark.parametrize(
    "summarizer",
    [feedback.QlibFactorExperiment2Feedback, feedback.QlibModelExperiment2Feedback],
)
def test_summarizer_copies_metric_configuration(summarizer):
    selected = ["valid_loss"]
    configured = summarizer(scen=Mock(), metrics=selected)
    selected.append("heldout_loss")
    assert configured.metrics == ["valid_loss"]
    assert summarizer(scen=Mock()).metrics == feedback.IMPORTANT_METRICS


@pytest.mark.parametrize(
    "summarizer",
    [feedback.QlibFactorExperiment2Feedback, feedback.QlibModelExperiment2Feedback],
)
def test_generated_feedback_uses_configured_metrics(summarizer, monkeypatch):
    scenario = Mock()
    scenario.get_scenario_all_desc.return_value = "Synthetic scenario."
    exp = SimpleNamespace()
    exp.hypothesis = SimpleNamespace(hypothesis="Synthetic hypothesis.", reason="Synthetic reason.")
    exp.result = results(["valid_loss", "heldout_loss"])
    exp.sub_workspace_list = [SimpleNamespace(file_dict={"model.py": "# synthetic"})]
    exp.stdout = "Synthetic training log."
    task = SyntheticTask()
    exp.sub_tasks = [task]
    previous = SimpleNamespace()
    previous.result = results(["valid_loss", "heldout_loss"])
    previous.sub_tasks = [task]
    previous.sub_workspace_list = [SimpleNamespace(file_dict={"model.py": "# synthetic"})]
    exp.based_experiments = [previous]
    trace = Mock()
    trace.get_sota_hypothesis_and_experiment.return_value = (exp.hypothesis, previous)
    backend = Mock()
    backend.build_messages_and_create_chat_completion.return_value = json.dumps({"Decision": False})
    monkeypatch.setattr(feedback, "APIBackend", lambda: backend)

    summarizer(scen=scenario, metrics=["valid_loss"]).generate_feedback(exp, trace)

    for call in backend.build_messages_and_create_chat_completion.call_args_list:
        assert "valid_loss" in call.kwargs["user_prompt"]
        assert "heldout_loss" not in call.kwargs["user_prompt"]
    assert list(exp.result.index) == ["valid_loss", "heldout_loss"]
    assert list(previous.result.index) == ["valid_loss", "heldout_loss"]
