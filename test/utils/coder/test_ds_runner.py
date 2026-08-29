import unittest
from types import SimpleNamespace

import pytest

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.scenarios.data_science.dev.runner.eval import DSRunnerFeedback


def build_feedback(score, acceptable=True):
    return CoSTEERMultiFeedback(
        [
            DSRunnerFeedback(
                execution="executed",
                return_checking="checked",
                code="code",
                final_decision=True,
                acceptable=acceptable,
                score=score,
            ),
        ],
    )


def build_runner(metric_direction):
    """Build a runner that only carries the scenario attribute `should_use_new_evo` reads."""
    runner = object.__new__(DSCoSTEERRunner)
    runner.scen = SimpleNamespace(metric_direction=metric_direction)
    return runner


@pytest.mark.offline
class DSCoSTEERRunnerScorePolicyTest(unittest.TestCase):
    """`should_use_new_evo` must treat a tie the same way for both metric directions."""

    def test_tie_is_not_an_improvement_when_minimizing(self):
        runner = build_runner(metric_direction=False)
        self.assertFalse(runner.should_use_new_evo(build_feedback(1.0), build_feedback(1.0)))

    def test_tie_is_not_an_improvement_when_maximizing(self):
        runner = build_runner(metric_direction=True)
        self.assertFalse(runner.should_use_new_evo(build_feedback(1.0), build_feedback(1.0)))

    def test_lower_score_is_an_improvement_when_minimizing(self):
        runner = build_runner(metric_direction=False)
        self.assertTrue(runner.should_use_new_evo(build_feedback(1.0), build_feedback(0.5)))

    def test_higher_score_is_not_an_improvement_when_minimizing(self):
        runner = build_runner(metric_direction=False)
        self.assertFalse(runner.should_use_new_evo(build_feedback(1.0), build_feedback(1.5)))

    def test_higher_score_is_an_improvement_when_maximizing(self):
        runner = build_runner(metric_direction=True)
        self.assertTrue(runner.should_use_new_evo(build_feedback(1.0), build_feedback(1.5)))

    def test_lower_score_is_not_an_improvement_when_maximizing(self):
        runner = build_runner(metric_direction=True)
        self.assertFalse(runner.should_use_new_evo(build_feedback(1.0), build_feedback(0.5)))

    def test_missing_base_feedback_is_accepted(self):
        for metric_direction in (True, False):
            runner = build_runner(metric_direction=metric_direction)
            self.assertTrue(runner.should_use_new_evo(None, build_feedback(1.0)))

    def test_missing_new_score_is_rejected(self):
        for metric_direction in (True, False):
            runner = build_runner(metric_direction=metric_direction)
            self.assertFalse(runner.should_use_new_evo(build_feedback(1.0), build_feedback(None)))

    def test_missing_base_score_is_accepted(self):
        for metric_direction in (True, False):
            runner = build_runner(metric_direction=metric_direction)
            self.assertTrue(runner.should_use_new_evo(build_feedback(None), build_feedback(1.0)))

    def test_unacceptable_feedback_is_rejected(self):
        runner = build_runner(metric_direction=False)
        self.assertFalse(
            runner.should_use_new_evo(build_feedback(1.0), build_feedback(0.5, acceptable=False)),
        )


if __name__ == "__main__":
    unittest.main()
