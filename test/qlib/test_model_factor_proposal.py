import pytest
from unittest.mock import Mock, patch

from rdagent.core.proposal import Trace, Hypothesis

# Proposal converters
from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesis2Experiment
from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesis2Experiment


# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def mixed_model_trace():
    trace = Trace(scen=Mock())

    # Mock tasks with names
    model_task = Mock()
    model_task.name = "model_task_1"

    factor_task = Mock()
    factor_task.name = "factor_task_1"

    trace.hist = [
        (Mock(sub_tasks=[model_task], hypothesis=Mock(action="model")), Mock()),
        (Mock(sub_tasks=[factor_task], hypothesis=Mock(action="factor")), Mock())
    ]
    return trace


@pytest.fixture
def mixed_factor_trace():
    trace = Trace(scen=Mock())

    # Mock tasks with factor_name
    factor_task = Mock()
    factor_task.factor_name = "factor_task_1"

    model_task = Mock()
    model_task.name = "model_task_1"

    trace.hist = [
        (Mock(sub_tasks=[factor_task], hypothesis=Mock(action="factor")), Mock()),
        (Mock(sub_tasks=[model_task], hypothesis=Mock(action="model")), Mock())
    ]
    return trace


# -------------------------
# Tests
# -------------------------
def test_model_proposal_import():
    from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesis2Experiment
    assert QlibModelHypothesis2Experiment is not None


def test_factor_proposal_import():
    from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesis2Experiment
    assert QlibFactorHypothesis2Experiment is not None


def test_model_filtering(mixed_model_trace):
    converter = QlibModelHypothesis2Experiment()
    hypothesis = Hypothesis(
        hypothesis="test",
        reason="r",
        concise_reason="cr",
        concise_observation="co",
        concise_justification="cj",
        concise_knowledge="ck"
    )

    # Patch template rendering to avoid Jinja2 errors
    with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
        context, ok = converter.prepare_context(hypothesis, mixed_model_trace)

    target_list = context.get("target_list", [])
    assert ok is True

    # Check that all target tasks have "model" in their name
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
        concise_knowledge="ck"
    )

    with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
        context, ok = converter.prepare_context(hypothesis, mixed_factor_trace)

    target_list = context.get("target_list", [])
    assert ok is True

    # Only factor tasks should remain
    factor_names = [getattr(task, "factor_name", "") for task in target_list]
    assert all("factor" in name for name in factor_names)


# -------------------------
# Code inspection tests
# -------------------------
@pytest.mark.parametrize(
    "converter_class, expected_type_str",
    [
        (QlibModelHypothesis2Experiment, "ModelExperiment"),
        (QlibFactorHypothesis2Experiment, "FactorExperiment"),
    ],
)
def test_code_inspection(converter_class, expected_type_str):
    import inspect
    source = inspect.getsource(converter_class.prepare_context)
    assert f"isinstance(t[0], {expected_type_str})" in source
