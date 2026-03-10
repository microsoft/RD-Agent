from __future__ import annotations

from pathlib import Path

import typer

from rdagent.crypto.btcusdt_15m_signal_writer_v1 import (
    BTCUSDT15mSignalWriterV1Config,
    generate_btcusdt_15m_signals_csv_v1,
)


def btcusdt_15m_signal_writer_v1(
    candles_csv_path: str = typer.Option(..., "--input", help="Input candles CSV with 'timestamp' and 'close'"),
    output_csv_path: str = typer.Option(..., "--output", help="Output CSV path for signals"),
    symbol: str = typer.Option("BTCUSDT", help="Trading symbol"),
    interval: str = typer.Option("15m", help="Bar interval"),
    fast_span: int = typer.Option(12, help="Fast EMA span"),
    slow_span: int = typer.Option(26, help="Slow EMA span"),
) -> str:
    """Write BTCUSDT 15m crossover signals (v1) to CSV."""

    cfg = BTCUSDT15mSignalWriterV1Config(
        symbol=symbol,
        interval=interval,
        fast_span=fast_span,
        slow_span=slow_span,
    )
    out = generate_btcusdt_15m_signals_csv_v1(
        candles_csv_path=Path(candles_csv_path),
        output_csv_path=Path(output_csv_path),
        config=cfg,
    )
    return str(out)
