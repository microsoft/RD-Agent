from __future__ import annotations

from pathlib import Path

import pandas as pd

from rdagent.crypto.btcusdt_15m_signal_writer_v1 import (
    BTCUSDT15mSignalWriterV1Config,
    compute_btcusdt_15m_signals_v1,
    generate_btcusdt_15m_signals_csv_v1,
)


def _make_candles(epoch_unit: str) -> pd.DataFrame:
    # Make a close series that forces a bull cross and then a bear cross.
    close = [1] * 10 + [10] * 10 + [1] * 10
    start = 1_700_000_000  # epoch seconds
    ts = [start + i * 900 for i in range(len(close))]  # 15m steps
    if epoch_unit == "ms":
        ts = [t * 1000 for t in ts]
    return pd.DataFrame({"timestamp": ts, "close": close})


def test_compute_signals_seconds_epoch() -> None:
    candles = _make_candles("s")
    cfg = BTCUSDT15mSignalWriterV1Config(fast_span=2, slow_span=5)
    out = compute_btcusdt_15m_signals_v1(candles, config=cfg)

    assert list(out.columns) == ["timestamp", "symbol", "interval", "signal", "version"]
    assert out["symbol"].nunique() == 1
    assert out["interval"].nunique() == 1
    assert out["version"].nunique() == 1
    assert 1 in set(out["signal"].tolist())
    assert -1 in set(out["signal"].tolist())


def test_compute_signals_milliseconds_epoch() -> None:
    candles = _make_candles("ms")
    cfg = BTCUSDT15mSignalWriterV1Config(fast_span=2, slow_span=5)
    out = compute_btcusdt_15m_signals_v1(candles, config=cfg)
    assert out["timestamp"].str.endswith("Z").all()


def test_generate_csv(tmp_path: Path) -> None:
    candles = _make_candles("s")
    candles_path = tmp_path / "candles.csv"
    out_path = tmp_path / "signals.csv"
    candles.to_csv(candles_path, index=False)

    cfg = BTCUSDT15mSignalWriterV1Config(fast_span=2, slow_span=5)
    wrote = generate_btcusdt_15m_signals_csv_v1(
        candles_csv_path=candles_path,
        output_csv_path=out_path,
        config=cfg,
    )
    assert wrote == out_path
    assert out_path.exists()

    signals = pd.read_csv(out_path)
    assert list(signals.columns) == ["timestamp", "symbol", "interval", "signal", "version"]
