import pandas as pd
import pytest

from rdagent.components.coder.factor_coder.eva_utils import (
    FactorDatetimeDailyEvaluator,
)


def _evaluate_datetimes(datetimes: pd.DatetimeIndex) -> tuple[str, bool]:
    index = pd.MultiIndex.from_product(
        [datetimes, ["instrument_1", "instrument_2"]],
        names=["datetime", "instrument"],
    )

    return _evaluate_index(index)


def _evaluate_index(index: pd.MultiIndex) -> tuple[str, bool]:
    gen_df = pd.DataFrame({"factor": range(len(index))}, index=index)
    evaluator = object.__new__(FactorDatetimeDailyEvaluator)
    evaluator._get_df = lambda _gt_implementation, _implementation: (  # noqa: SLF001
        None,
        gen_df,
    )
    return evaluator.evaluate(None, None)


@pytest.mark.offline
@pytest.mark.parametrize("freq", ["1min", "s", "30min", "h"])
def test_datetime_daily_evaluator_rejects_subdaily_frequency(freq: str) -> None:
    message, is_daily = _evaluate_datetimes(
        pd.date_range("2024-01-01", periods=3, freq=freq),
    )

    assert is_daily is False, message


@pytest.mark.offline
def test_datetime_daily_evaluator_allows_daily_multi_instrument_data() -> None:
    datetimes = pd.date_range("2024-01-01", periods=3, freq="D")

    message, is_daily = _evaluate_datetimes(datetimes)

    assert is_daily is True, message


@pytest.mark.offline
def test_datetime_daily_evaluator_rejects_mixed_daily_and_subdaily_frequency() -> None:
    datetimes = pd.DatetimeIndex(
        [
            "2024-01-01 00:00:00",
            "2024-01-02 00:00:00",
            "2024-01-02 01:00:00",
        ],
    )

    message, is_daily = _evaluate_datetimes(datetimes)

    assert is_daily is False, message


@pytest.mark.offline
def test_datetime_daily_evaluator_allows_daily_calendar_gaps() -> None:
    datetimes = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])

    message, is_daily = _evaluate_datetimes(datetimes)

    assert is_daily is True, message


@pytest.mark.offline
def test_datetime_daily_evaluator_allows_daily_data_across_dst() -> None:
    datetimes = pd.date_range(
        "2024-03-09",
        periods=3,
        freq="D",
        tz="America/New_York",
    )

    message, is_daily = _evaluate_datetimes(datetimes)

    assert is_daily is True, message


@pytest.mark.offline
def test_datetime_daily_evaluator_rejects_subdaily_instrument_first_index() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("instrument_1", "2024-01-01 00:00:00"),
            ("instrument_1", "2024-01-02 00:00:00"),
            ("instrument_2", "2024-01-01 23:00:00"),
            ("instrument_2", "2024-01-02 23:00:00"),
        ],
        names=["instrument", "datetime"],
    )

    message, is_daily = _evaluate_index(index)

    assert is_daily is False, message


@pytest.mark.offline
def test_datetime_daily_evaluator_rejects_missing_datetimes() -> None:
    message, is_daily = _evaluate_datetimes(
        pd.DatetimeIndex(["2024-01-01", pd.NaT]),
    )

    assert is_daily is False, message
