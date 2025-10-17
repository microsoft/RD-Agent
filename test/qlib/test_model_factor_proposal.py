from unittest.mock import Mock, patch

import pytest

from rdagent.core.proposal import Hypothesis, Trace
from rdagent.scenarios.qlib.proposal.factor_proposal import (
    QlibFactorHypothesis2Experiment,
)
from rdagent.scenarios.qlib.proposal.model_proposal import (
    QlibModelHypothesis2Experiment,
)


@pytest.fixture
def mixed_model_trace():
    trace = Trace(scen=Mock())
    model_task = Mock()
    model_task.name = "model_task_1"
    factor_task = Mock()
    factor_task.name = "factor_task_1"
    trace.hist = [
        (Mock(sub_tasks=[model_task], hypothesis=Mock(action="model")), Mock()),
        (Mock(sub_tasks=[factor_task], hypothesis=Mock(action="factor")), Mock()),
    ]
    return trace


@pytest.fixture
def mixed_factor_trace():
    trace = Trace(scen=Mock())
    factor_task = Mock()
    factor_task.factor_name = "factor_task_1"
    model_task = Mock()
    model_task.name = "model_task_1"
    trace.hist = [
        (Mock(sub_tasks=[factor_task], hypothesis=Mock(action="factor")), Mock()),
        (Mock(sub_tasks=[model_task], hypothesis=Mock(action="model")), Mock()),
    ]
    return trace


def test_model_proposal_import():
    assert QlibModelHypothesis2Experiment is not None


def test_factor_proposal_import():
    assert QlibFactorHypothesis2Experiment is not None


def test_model_filtering(mixed_model_trace):
    converter = QlibModelHypothesis2Experiment()
    hypothesis = Hypothesis(
        hypothesis="test",
        reason="r",
        concise_reason="cr",
        concise_observation="co",
        concise_justification="cj",
        concise_knowledge="ck",
    )
    with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
        context, ok = converter.prepare_context(hypothesis, mixed_model_trace)

    target_list = context.get("target_list", [])
    assert ok is True
    names = [getattr(task, "name", "") for task in target_list]
    assert all("model" in name for name in names)


def test_factor_filtering(mixed_factor_trace):
    converter = QlibFactorHypothesis2Experiment()
    hypothesis = Hypothesis(
        hypothesis="test",
        reason="r",
        concise_reason="cr",
        concise_observation="co",
        concise_justification="cj",
        concise_knowledge="ck",
    )
    with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
        context, ok = converter.prepare_context(hypothesis, mixed_factor_trace)

    target_list = context.get("target_list", [])
    assert ok is True
    factor_names = [getattr(task, "factor_name", "") for task in target_list]
    assert all("factor" in name for name in factor_names)


@pytest.mark.parametrize(
    "converter_class, trace_fixture, expected_type",
    [
        (QlibModelHypothesis2Experiment, "mixed_model_trace", "ModelExperiment"),
        (QlibFactorHypothesis2Experiment, "mixed_factor_trace", "FactorExperiment"),
    ],
)
def test_code_inspection(converter_class, trace_fixture, request, expected_type):
    converter = converter_class()
    trace = request.getfixturevalue(trace_fixture)
    hypothesis = Hypothesis(
        hypothesis="test",
        reason="r",
        concise_reason="cr",
        concise_observation="co",
        concise_justification="cj",
        concise_knowledge="ck",
    )
    with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
        context, ok = converter.prepare_context(hypothesis, trace)

    target_list = context.get("target_list", [])
    assert ok is True
    if target_list:
        assert target_list[0].__class__.__name__ == expected_type
