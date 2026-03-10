from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BTCUSDT_15M_SIGNAL_WRITER_VERSION = "v1"


@dataclass(frozen=True)
class BTCUSDT15mSignalWriterV1Config:
    symbol: str = "BTCUSDT"
    interval: str = "15m"
    fast_span: int = 12
    slow_span: int = 26


def _coerce_timestamp_to_utc(ts: pd.Series) -> pd.Series:
    """Coerce timestamps to timezone-aware UTC pandas Timestamps.

    Accepts:
    - datetime-like strings
    - epoch seconds
    - epoch milliseconds
    """

    if pd.api.types.is_datetime64_any_dtype(ts):
        out = pd.to_datetime(ts, utc=True)
    else:
        s = pd.to_numeric(ts, errors="coerce")
        if s.isna().any():
            raise ValueError("timestamp contains non-parseable values")
        unit = "ms" if s.max() >= 1_000_000_000_000 else "s"
        out = pd.to_datetime(s, unit=unit, utc=True)
    if out.isna().any():
        raise ValueError("timestamp contains NaT values after parsing")
    return out


def compute_btcusdt_15m_signals_v1(
    candles: pd.DataFrame,
    *,
    config: BTCUSDT15mSignalWriterV1Config | None = None,
) -> pd.DataFrame:
    """Compute BTCUSDT 15m crossover signals.

    Input schema:
    - timestamp: datetime-like, epoch seconds, or epoch milliseconds
    - close: numeric

    Output schema (CSV-ready):
    - timestamp: UTC ISO timestamp
    - symbol: trading pair (default BTCUSDT)
    - interval: bar interval (default 15m)
    - signal: 1 (bull cross), -1 (bear cross), 0 (no signal)
    - version: writer version
    """

    cfg = config or BTCUSDT15mSignalWriterV1Config()

    if "timestamp" not in candles.columns or "close" not in candles.columns:
        raise ValueError("candles must include 'timestamp' and 'close' columns")

    df = candles[["timestamp", "close"]].copy()
    df["timestamp"] = _coerce_timestamp_to_utc(df["timestamp"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df["close"].isna().any():
        raise ValueError("close contains non-numeric values")

    df = df.sort_values("timestamp").reset_index(drop=True)
    if cfg.fast_span <= 0 or cfg.slow_span <= 0:
        raise ValueError("fast_span and slow_span must be positive")
    if cfg.fast_span >= cfg.slow_span:
        raise ValueError("fast_span must be < slow_span")

    ema_fast = df["close"].ewm(span=cfg.fast_span, adjust=False).mean()
    ema_slow = df["close"].ewm(span=cfg.slow_span, adjust=False).mean()

    bull = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    bear = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

    signal = pd.Series(0, index=df.index, dtype="int64")
    signal[bull] = 1
    signal[bear] = -1

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": cfg.symbol,
            "interval": cfg.interval,
            "signal": signal,
            "version": BTCUSDT_15M_SIGNAL_WRITER_VERSION,
        }
    )
    return out


def generate_btcusdt_15m_signals_csv_v1(
    *,
    candles_csv_path: str | Path,
    output_csv_path: str | Path,
    config: BTCUSDT15mSignalWriterV1Config | None = None,
) -> Path:
    """Read candles from CSV, compute signals, and write a signal CSV."""

    candles_csv_path = Path(candles_csv_path)
    output_csv_path = Path(output_csv_path)

    candles = pd.read_csv(candles_csv_path)
    signals = compute_btcusdt_15m_signals_v1(candles, config=config)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_csv_path, index=False)
    return output_csv_path
