import ast
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from rdagent.log.storage import FileStorage

APP_PATH = Path(__file__).parents[3] / "rdagent/log/ui/app.py"
CHART_TAG = "Quantitative Backtesting Chart"


def _function(name: str) -> ast.FunctionDef:
    """Return a top level function of the Streamlit app.

    ``app.py`` is an entry point that runs code at import time, so its source is inspected
    statically instead of being imported.
    """
    for node in ast.parse(APP_PATH.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {APP_PATH}")


def _report_figure_arguments() -> list[str]:
    """Unparse the argument of every `report_figure(...)` call in `feedback_window`."""
    return [
        ast.unparse(node.args[0])
        for node in ast.walk(_function("feedback_window"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "report_figure"
    ]


@pytest.mark.offline
def test_latest_backtest_chart_returns_the_last_message() -> None:
    returns = [node for node in ast.walk(_function("latest_backtest_chart")) if isinstance(node, ast.Return)]

    assert len(returns) == 1
    assert ast.unparse(returns[0].value) == "chart_msgs[-1]"


@pytest.mark.offline
def test_feedback_window_renders_the_latest_chart() -> None:
    # the chart entry used to be indexed with [0], which is the oldest chart of the round
    assert _report_figure_arguments() == ["latest_backtest_chart(fbr).content"]


@pytest.mark.offline
def test_iter_msg_yields_messages_in_ascending_timestamp_order(tmp_path: Path) -> None:
    """`latest_backtest_chart` relies on the last yielded entry being the newest.

    A round logs the baseline experiment before it logs its own result, so the last chart
    under the tag is the one that belongs to the round.
    """
    storage = FileStorage(tmp_path / "trace")
    early = datetime(2026, 9, 20, 3, 2, 28, tzinfo=timezone.utc)
    late = datetime(2026, 9, 20, 3, 3, 8, tzinfo=timezone.utc)
    # the earlier chart is the baseline run, the later one is the result of the round.
    # they are written out of order to show that the yielded order follows the timestamps.
    storage.log(pd.DataFrame({"return": [0.0, -0.1]}), tag=CHART_TAG, timestamp=late)
    storage.log(pd.DataFrame({"return": [0.0, 0.1]}), tag=CHART_TAG, timestamp=early)

    messages = list(storage.iter_msg())

    assert [msg.timestamp for msg in messages] == [early, late]
    assert messages[0].content["return"].iloc[-1] == 0.1
    assert messages[-1].content["return"].iloc[-1] == -0.1
