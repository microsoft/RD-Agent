"""Tests for the metric extraction that feeds the fin_quant bandit scheduler (see #1451).

Only ``experiment.result`` is touched, so a ``SimpleNamespace`` carrying a Series with the real Qlib key names is a
faithful stand-in; no Qlib, Docker or LLM is needed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from rdagent.scenarios.qlib.developer.feedback import IMPORTANT_METRICS
from rdagent.scenarios.qlib.metrics import (
    ARR_KEY,
    IC_KEY,
    ICIR_KEY,
    IR_KEY,
    MDD_KEY,
    RANK_IC_KEY,
    RANK_ICIR_KEY,
)
from rdagent.scenarios.qlib.proposal.bandit import (
    EnvController,
    extract_metrics_from_experiment,
)

# The exact keys Qlib logs (SigAnaRecord and PortAnaRecord), as read back from qlib_res.csv by QlibFBWorkspace.
QLIB_RESULT = pd.Series(
    {
        "IC": 0.05,
        "ICIR": 0.4,
        "Rank IC": 0.06,
        "Rank ICIR": 0.5,
        "1day.excess_return_with_cost.annualized_return": 0.12,
        "1day.excess_return_with_cost.information_ratio": 1.1,
        "1day.excess_return_with_cost.max_drawdown": -0.08,
    }
)

LOGGER = "rdagent.scenarios.qlib.proposal.bandit.logger"


def _experiment(result):
    return SimpleNamespace(result=result)


@pytest.mark.offline
def test_metric_keys_match_qlib_output():
    for key in (IC_KEY, ICIR_KEY, RANK_IC_KEY, RANK_ICIR_KEY, ARR_KEY, IR_KEY, MDD_KEY):
        assert key == key.strip()
        assert key in QLIB_RESULT.index, key
    assert set(IMPORTANT_METRICS) <= set(QLIB_RESULT.index)


@pytest.mark.offline
def test_extract_reads_every_metric():
    m = extract_metrics_from_experiment(_experiment(QLIB_RESULT))

    assert m.ic == pytest.approx(0.05)
    assert m.arr == pytest.approx(0.12)
    assert m.ir == pytest.approx(1.1)
    assert m.mdd == pytest.approx(-0.08)
    assert m.calmar == pytest.approx(0.12 / 0.08)

    vec = m.as_vector()
    assert vec.shape == (8,)
    assert vec[4] == pytest.approx(0.12)
    assert vec[6] == pytest.approx(0.08)  # -mdd, so a deeper drawdown lowers the slot
    assert vec[7] > 0


@pytest.mark.offline
def test_missing_key_warns_and_defaults():
    partial = QLIB_RESULT.drop(ARR_KEY)
    with patch(LOGGER) as logger:
        m = extract_metrics_from_experiment(_experiment(partial))

    assert m.arr == 0.0
    assert m.calmar == 0.0
    assert m.ic == pytest.approx(0.05)
    warnings = [call.args[0] for call in logger.warning.call_args_list]
    assert any(ARR_KEY in msg for msg in warnings), warnings


@pytest.mark.offline
def test_missing_drawdown_is_neutral():
    partial = QLIB_RESULT.drop(MDD_KEY)
    with patch(LOGGER):
        m = extract_metrics_from_experiment(_experiment(partial))

    assert m.mdd == 0.0
    assert m.calmar == 0.0
    assert m.as_vector()[6] == 0.0


@pytest.mark.offline
def test_failed_run_gives_zero_vector():
    with patch(LOGGER):
        m = extract_metrics_from_experiment(_experiment(None))
    assert not m.as_vector().any()


@pytest.mark.offline
def test_arr_moves_the_reward():
    controller = EnvController()
    base = extract_metrics_from_experiment(_experiment(QLIB_RESULT))

    doubled = QLIB_RESULT.copy()
    doubled[ARR_KEY] = 0.24
    better = extract_metrics_from_experiment(_experiment(doubled))

    # Both the ARR slot (weight 0.25) and the derived ratio (weight 0.2) respond, so 0.45 of the weight is live.
    expected_gain = 0.25 * (0.24 - 0.12) + 0.2 * ((0.24 - 0.12) / 0.08)
    assert controller.reward(better) - controller.reward(base) == pytest.approx(expected_gain)
