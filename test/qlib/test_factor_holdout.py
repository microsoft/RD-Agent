from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner

IMPORTANT_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.max_drawdown",
]


def _result(value: float) -> pd.Series:
    return pd.Series(dict.fromkeys(IMPORTANT_METRICS, value))


def test_factor_search_uses_validation_as_qlib_evaluation_segment():
    props = SimpleNamespace(
        train_start="2008-01-01",
        train_end="2014-12-31",
        valid_start="2015-01-01",
        valid_end="2016-12-31",
        test_start="2017-01-01",
        test_end="2020-08-01",
    )
    exp = Mock(base_features={"factor": "expression"})
    runner = object.__new__(QlibFactorRunner)

    with patch("rdagent.scenarios.qlib.developer.factor_runner.FactorBasePropSetting", return_value=props):
        search_env = runner._get_run_env(exp, use_holdout=False)
        holdout_env = runner._get_run_env(exp, use_holdout=True)

    assert search_env["test_start"] == props.valid_start
    assert search_env["test_end"] == props.valid_end
    assert holdout_env["test_start"] == props.test_start
    assert holdout_env["test_end"] == props.test_end


def test_final_holdout_does_not_replace_search_result():
    search_result = _result(0.02)
    holdout_result = _result(0.99)
    exp = Mock(result=search_result)
    runner = object.__new__(QlibFactorRunner)
    runner._run_experiment = Mock(return_value=(holdout_result, "holdout stdout"))

    result = runner.evaluate_holdout(exp)

    assert result is holdout_result
    assert exp.result is search_result
    assert exp.holdout_result is holdout_result
    assert exp.holdout_stdout == "holdout stdout"

