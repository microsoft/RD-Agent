from types import SimpleNamespace

import pytest

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.scenarios.data_science.dev.runner.eval import DSRunnerFeedback

pytestmark = pytest.mark.offline


def make_feedback(score, acceptable=True):
    return CoSTEERMultiFeedback(
        [
            DSRunnerFeedback(
                execution="ok",
                return_checking="ok",
                code="ok",
                final_decision=acceptable,
                acceptable=acceptable,
                score=score,
            )
        ]
    )


@pytest.mark.parametrize("higher_is_better", [True, False], ids=["maximize", "minimize"])
@pytest.mark.parametrize(
    "base_score,new_score,maximize_result,minimize_result",
    [
        (1.0, 1.0, False, False),
        (1.0, 2.0, True, False),
        (1.0, 0.5, False, True),
        (None, 1.0, True, True),
        (1.0, None, False, False),
        (None, None, False, False),
    ],
    ids=["tie", "higher", "lower", "missing-base-score", "missing-new-score", "both-scores-missing"],
)
def test_runner_selects_only_strict_score_improvements(
    higher_is_better, base_score, new_score, maximize_result, minimize_result
):
    runner = object.__new__(DSCoSTEERRunner)
    runner.scen = SimpleNamespace(metric_direction=higher_is_better)
    expected = maximize_result if higher_is_better else minimize_result
    assert runner.should_use_new_evo(make_feedback(base_score), make_feedback(new_score)) is expected


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_runner_accepts_first_acceptable_feedback(higher_is_better):
    runner = object.__new__(DSCoSTEERRunner)
    runner.scen = SimpleNamespace(metric_direction=higher_is_better)
    assert runner.should_use_new_evo(None, make_feedback(1.0)) is True


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_runner_rejects_unacceptable_feedback_even_without_a_baseline(higher_is_better):
    runner = object.__new__(DSCoSTEERRunner)
    runner.scen = SimpleNamespace(metric_direction=higher_is_better)
    assert runner.should_use_new_evo(None, make_feedback(1.0, acceptable=False)) is False
