"""Regression tests for issue #1407.

The factor coder occasionally generates ML-based factors whose
implementation wraps model construction or training inside a nested loop
over instruments and time.  On a realistic 5K-instrument / 1K-day A-share
panel that produces O(instruments * days) training iterations and the run
hangs for hours.  ``detect_per_instrument_training_antipattern`` catches
the pattern statically so CoSTEER can surface critic feedback before
spending the wall-clock cost of executing the offending code.
"""

from __future__ import annotations

import unittest

import pytest

from rdagent.components.coder.factor_coder.evaluators import (
    detect_per_instrument_training_antipattern,
)


@pytest.mark.offline
class DetectPerInstrumentTrainingAntipatternTest(unittest.TestCase):

    def test_lstm_per_instrument_per_day_loop_is_flagged(self) -> None:
        code = """
import pandas as pd
import torch
import torch.nn as nn


def calculate_lstm_factor(df):
    instruments = df.index.get_level_values('instrument').unique()
    trading_days = df.index.get_level_values('datetime').unique()
    result = pd.Series(index=df.index, dtype=float)
    for instrument in instruments:
        for day in trading_days:
            model = nn.LSTM(input_size=5, hidden_size=8, num_layers=1)
            optimizer = torch.optim.Adam(model.parameters())
            # ... train per (instrument, day) ...
            model.fit(df.loc[(slice(None, day), instrument)])
            pred = model.predict(df.loc[(day, instrument)])
            result.loc[(day, instrument)] = pred
    return result
"""

        feedback = detect_per_instrument_training_antipattern(code)
        self.assertIsNotNone(feedback)
        self.assertIn("anti-pattern", feedback)
        self.assertIn("#1407", feedback)

    def test_random_forest_per_stock_retraining_is_flagged(self) -> None:
        code = """
from sklearn.ensemble import RandomForestRegressor


def calculate_rf_factor(df):
    for stock_code in df.index.get_level_values('instrument').unique():
        for day in df.index.get_level_values('datetime').unique():
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(features, target)
            preds.append(rf.predict(features))
    return preds
"""

        feedback = detect_per_instrument_training_antipattern(code)
        self.assertIsNotNone(feedback)

    def test_xgboost_per_ticker_loop_is_flagged(self) -> None:
        code = """
import xgboost as xgb


def calculate_factor(df):
    for ticker in tickers:
        for d in days:
            booster = xgb.XGBRegressor(n_estimators=200)
            booster.fit(X_train, y_train)
            out[ticker, d] = booster.predict(X_test)[0]
    return out
"""

        feedback = detect_per_instrument_training_antipattern(code)
        self.assertIsNotNone(feedback)

    def test_panel_level_single_fit_is_allowed(self) -> None:
        # The recommended pattern -- one fit on the full panel, then batch
        # predict -- must not be flagged.
        code = """
from sklearn.ensemble import RandomForestRegressor


def calculate_factor(panel_df, X, y):
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, y)
    panel_df['factor'] = model.predict(X)
    return panel_df
"""

        self.assertIsNone(detect_per_instrument_training_antipattern(code))

    def test_nested_loop_without_training_call_is_allowed(self) -> None:
        # Nested iteration over instruments and dates with only statistical
        # operations (no .fit / no estimator constructor) must not be flagged.
        code = """
def calculate_momentum(df):
    out = {}
    for instrument in df.index.get_level_values('instrument').unique():
        for day in df.index.get_level_values('datetime').unique():
            out[(day, instrument)] = df.loc[(day, instrument), 'close'].pct_change(20).mean()
    return out
"""

        self.assertIsNone(detect_per_instrument_training_antipattern(code))

    def test_groupby_rolling_apply_is_allowed(self) -> None:
        code = """
def calculate_rolling_factor(df):
    return (
        df.groupby(level='instrument')['close']
          .rolling(20)
          .apply(lambda x: x.mean())
    )
"""

        self.assertIsNone(detect_per_instrument_training_antipattern(code))

    def test_syntax_error_returns_none(self) -> None:
        # Syntax-broken code is handled by the normal execution path -- the
        # detector must not raise on it.
        self.assertIsNone(detect_per_instrument_training_antipattern("def broken("))

    def test_empty_code_returns_none(self) -> None:
        self.assertIsNone(detect_per_instrument_training_antipattern(""))


if __name__ == "__main__":
    unittest.main()
