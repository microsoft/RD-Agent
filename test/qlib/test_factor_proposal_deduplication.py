import json
from unittest.mock import Mock

import pytest

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.exception import CoderError
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.proposal.factor_proposal import (
    QlibFactorHypothesis2Experiment,
)

pytestmark = pytest.mark.offline


def factor_task(name):
    return FactorTask(factor_name=name, factor_description="existing factor", factor_formulation="$close", variables={})


def convert(names, history):
    trace = Trace(scen=Mock())
    trace.hist = history
    hypothesis = Hypothesis(
        hypothesis="test",
        reason="r",
        concise_reason="cr",
        concise_observation="co",
        concise_justification="cj",
        concise_knowledge="ck",
    )
    response = json.dumps(
        {name: {"description": "candidate factor", "formulation": "$open", "variables": {}} for name in names}
    )
    result = QlibFactorHypothesis2Experiment().convert_response(response, hypothesis, trace)
    assert result.hypothesis is hypothesis
    return result


@pytest.mark.parametrize(
    "proposed, expected",
    [
        (["accepted"], []),
        (["accepted", "new_a", "new_b"], ["new_a", "new_b"]),
        (["new_a", "new_b"], ["new_a", "new_b"]),
    ],
    ids=["all-duplicate", "mixed", "all-new"],
)
def test_filters_accepted_names_before_creating_workspaces(proposed, expected):
    accepted = QlibFactorExperiment([factor_task("accepted")])
    result = convert(proposed, [(accepted, True)])

    assert [task.factor_name for task in result.sub_tasks] == expected
    assert result.sub_workspace_list == [None] * len(expected)
    assert result.based_experiments[1:] == [accepted]
    assert isinstance(result.based_experiments[0], QlibFactorExperiment)
    assert result.based_experiments[0].sub_tasks == []
    assert accepted.sub_tasks[0].factor_formulation == "$close"


def test_rejected_factor_name_can_be_retried():
    rejected = QlibFactorExperiment([factor_task("retry")])
    result = convert(["retry"], [(rejected, False)])

    assert [task.factor_name for task in result.sub_tasks] == ["retry"]
    assert result.sub_workspace_list == [None]
    assert len(result.based_experiments) == 1


def test_model_history_does_not_participate_in_factor_name_filtering():
    accepted = QlibFactorExperiment([factor_task("accepted")])
    model = QlibModelExperiment(sub_tasks=[Mock(spec=["name"])])
    result = convert(["accepted", "new"], [(accepted, True), (model, True)])

    assert [task.factor_name for task in result.sub_tasks] == ["new"]
    assert result.sub_workspace_list == [None]
    assert result.based_experiments[1:] == [accepted, model]


def test_first_proposal_keeps_all_tasks_in_order():
    result = convert(["new_b", "new_a"], [])

    assert [task.factor_name for task in result.sub_tasks] == ["new_b", "new_a"]
    assert result.sub_workspace_list == [None, None]
    assert len(result.based_experiments) == 1


def test_all_duplicate_proposal_uses_existing_empty_coder_failure():
    accepted = QlibFactorExperiment([factor_task("accepted")])
    result = convert(["accepted"], [(accepted, True)])
    evolving_item = EvolvingItem.from_experiment(result)

    assert evolving_item.sub_tasks == []
    assert evolving_item.sub_workspace_list == []
    with pytest.raises(CoderError, match="All tasks are failed"):
        CoSTEER._exp_postprocess_by_feedback(None, evolving_item, CoSTEERMultiFeedback([]))
