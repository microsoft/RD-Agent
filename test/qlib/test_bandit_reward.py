"""Offline regression tests for the drawdown component of the bandit reward."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rdagent.scenarios.qlib.proposal.bandit import (
    EnvController,
    Metrics,
    extract_metrics_from_experiment,
)

pytestmark = pytest.mark.offline


def drawdown_metrics(mdd):
    # Keep every other metric, including the return/drawdown ratio, fixed at zero.
    experiment = SimpleNamespace(
        result={
            "IC": 0.0,
            "ICIR": 0.0,
            "Rank IC": 0.0,
            "Rank ICIR": 0.0,
            "1day.excess_return_with_cost.annualized_return": 0.0,
            "1day.excess_return_with_cost.information_ratio": 0.0,
            "1day.excess_return_with_cost.max_drawdown": mdd,
        }
    )
    return extract_metrics_from_experiment(experiment)


def test_deeper_drawdown_lowers_default_reward():
    controller = EnvController()
    rewards = [controller.reward(drawdown_metrics(mdd)) for mdd in (0.0, -0.1, -0.3)]

    assert rewards[0] == 0.0
    assert rewards[0] > rewards[1] > rewards[2]
    assert rewards == pytest.approx([0.0, -0.01, -0.03])


@pytest.mark.parametrize("arm", ["factor", "model"])
def test_record_uses_penalty_without_changing_context(arm, monkeypatch):
    controller = EnvController()
    metrics = drawdown_metrics(-0.3)
    expected_context = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0])
    update = Mock(wraps=controller.bandit.update)
    next_arm = Mock(wraps=controller.bandit.next_arm)
    monkeypatch.setattr(controller.bandit, "update", update)
    monkeypatch.setattr(controller.bandit, "next_arm", next_arm)

    controller.record(metrics, arm)
    controller.decide(metrics)

    update.assert_called_once()
    recorded_arm, context, reward = update.call_args.args
    assert recorded_arm == arm
    np.testing.assert_array_equal(context, expected_context)
    assert reward == pytest.approx(-0.03)
    assert controller.bandit.mean[arm][6] < 0.0
    next_arm.assert_called_once()
    np.testing.assert_array_equal(next_arm.call_args.args[0], expected_context)
    np.testing.assert_array_equal(metrics.as_vector(), expected_context)


@pytest.mark.parametrize(
    "field, index, weight",
    [
        ("ic", 0, 0.1),
        ("icir", 1, 0.1),
        ("rank_ic", 2, 0.05),
        ("rank_icir", 3, 0.05),
        ("arr", 4, 0.25),
        ("ir", 5, 0.15),
        ("sharpe", 7, 0.2),
    ],
)
def test_other_reward_components_and_context_are_unchanged(field, index, weight):
    metrics = Metrics(**{field: 0.4})
    expected_context = np.zeros(8)
    expected_context[index] = 0.4

    np.testing.assert_array_equal(metrics.as_vector(), expected_context)
    assert EnvController().reward(metrics) == pytest.approx(weight * 0.4)


def test_explicit_weights_keep_their_existing_meaning():
    weights = (0.1, 0.1, 0.05, 0.05, 0.25, 0.15, 0.1, 0.2)
    controller = EnvController(weights=weights)

    np.testing.assert_array_equal(controller.weights, weights)
    assert controller.reward(drawdown_metrics(-0.3)) == pytest.approx(0.03)
