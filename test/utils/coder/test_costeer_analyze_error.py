from types import SimpleNamespace

import pytest

from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERRAGStrategyV2,
)
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode,
)

ROWS_ERROR = "The source dataframe and the ground truth dataframe have different rows count."
TOLERANCE_ERROR = "Some values differ by more than the tolerance of 1e-6."
EXECUTION_FEEDBACK = 'File "factor.py", line 3, in <module>\n    x = 1 / 0\nZeroDivisionError: division by zero'
EXECUTION_ERROR = "ErrorType: ZeroDivisionError\nError line: x = 1 / 0"


def _error(content: str) -> UndirectedNode:
    return UndirectedNode(content=content, label="error")


def _strategy(*nodes: UndirectedNode) -> CoSTEERRAGStrategyV2:
    graph = UndirectedGraph()
    graph.nodes = {node.id: node for node in nodes}
    strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
    strategy.knowledgebase = SimpleNamespace(graph=graph)
    return strategy


@pytest.mark.offline
@pytest.mark.parametrize(
    ("feedback", "feedback_type", "content"),
    [
        (ROWS_ERROR, "value", ROWS_ERROR),
        (EXECUTION_FEEDBACK, "execution", EXECUTION_ERROR),
        ("Execution timed out after 600 seconds.", "execution", "Undefined Error"),
    ],
    ids=["value", "execution", "undefined"],
)
def test_analyze_error_returns_matched_node_once(feedback: str, feedback_type: str, content: str) -> None:
    matched = _error(content)
    strategy = _strategy(_error("A different previous error."), matched)
    assert strategy.analyze_error(feedback, feedback_type=feedback_type) == [matched]


@pytest.mark.offline
@pytest.mark.parametrize("graph_order", ["parsed", "reversed"])
def test_analyze_error_orders_matched_nodes_by_feedback(graph_order: str) -> None:
    rows, tolerance = _error(ROWS_ERROR), _error(TOLERANCE_ERROR)
    strategy = _strategy(rows, tolerance) if graph_order == "parsed" else _strategy(tolerance, rows)
    feedback = f"{ROWS_ERROR}\n{TOLERANCE_ERROR}\n{ROWS_ERROR}"
    assert strategy.analyze_error(feedback, feedback_type="value") == [rows, tolerance]


@pytest.mark.offline
def test_analyze_error_keeps_unmatched_error_in_order() -> None:
    rows = _error(ROWS_ERROR)
    strategy = _strategy(_error("A different previous error."), rows)
    assert strategy.analyze_error(f"{TOLERANCE_ERROR}\n{ROWS_ERROR}", feedback_type="value") == [TOLERANCE_ERROR, rows]
