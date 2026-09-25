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
EXECUTION_FEEDBACK = (
    'Traceback (most recent call last):\n  File "factor.py", line 3, in <module>\n'
    "    x = 1 / 0\nZeroDivisionError: division by zero"
)
EXECUTION_ERROR = "ErrorType: ZeroDivisionError\nError line: x = 1 / 0"
UNPARSED_FEEDBACK = "Execution timed out after 600 seconds."


def _strategy(error_nodes: list[UndirectedNode]) -> CoSTEERRAGStrategyV2:
    graph = UndirectedGraph()
    graph.nodes = {node.id: node for node in error_nodes}
    strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
    strategy.knowledgebase = SimpleNamespace(graph=graph)
    return strategy


@pytest.mark.offline
@pytest.mark.parametrize("matched_first", [True, False], ids=["matched_first", "matched_last"])
@pytest.mark.parametrize(
    ("feedback", "feedback_type", "error_content"),
    [
        pytest.param(ROWS_ERROR, "value", ROWS_ERROR, id="value"),
        pytest.param(EXECUTION_FEEDBACK, "execution", EXECUTION_ERROR, id="execution"),
    ],
)
def test_analyze_error_returns_matched_node_once(
    feedback: str, feedback_type: str, error_content: str, matched_first: bool
) -> None:
    matched = UndirectedNode(content=error_content, label="error")
    unrelated = UndirectedNode(content="A different previous error.", label="error")
    nodes = [matched, unrelated] if matched_first else [unrelated, matched]

    result = _strategy(nodes).analyze_error(feedback, feedback_type=feedback_type)

    assert len(result) == 1
    assert result[0] is matched


@pytest.mark.offline
@pytest.mark.parametrize("matched_first", [True, False], ids=["matched_first", "matched_last"])
def test_analyze_error_keeps_parsed_order(matched_first: bool) -> None:
    matched = UndirectedNode(content=ROWS_ERROR, label="error")
    unrelated = UndirectedNode(content="A different previous error.", label="error")
    nodes = [matched, unrelated] if matched_first else [unrelated, matched]
    strategy = _strategy(nodes)

    assert strategy.analyze_error(f"{ROWS_ERROR}\n{TOLERANCE_ERROR}", feedback_type="value") == [
        matched,
        TOLERANCE_ERROR,
    ]
    assert strategy.analyze_error(f"{TOLERANCE_ERROR}\n{ROWS_ERROR}", feedback_type="value") == [
        TOLERANCE_ERROR,
        matched,
    ]


@pytest.mark.offline
@pytest.mark.parametrize("matched_first", [True, False], ids=["matched_first", "matched_last"])
def test_analyze_error_returns_undefined_error_node_once(matched_first: bool) -> None:
    matched = UndirectedNode(content="Undefined Error", label="error")
    unrelated = UndirectedNode(content="A different previous error.", label="error")
    nodes = [matched, unrelated] if matched_first else [unrelated, matched]

    result = _strategy(nodes).analyze_error(UNPARSED_FEEDBACK, feedback_type="execution")

    assert len(result) == 1
    assert result[0] is matched


@pytest.mark.offline
@pytest.mark.parametrize("graph_order", ["parsed", "reversed"])
def test_analyze_error_orders_matched_nodes_by_feedback(graph_order: str) -> None:
    rows = UndirectedNode(content=ROWS_ERROR, label="error")
    tolerance = UndirectedNode(content=TOLERANCE_ERROR, label="error")
    nodes = [rows, tolerance] if graph_order == "parsed" else [tolerance, rows]

    result = _strategy(nodes).analyze_error(f"{ROWS_ERROR}\n{TOLERANCE_ERROR}", feedback_type="value")

    assert result == [rows, tolerance]


@pytest.mark.offline
def test_analyze_error_reports_repeated_error_once() -> None:
    matched = UndirectedNode(content=ROWS_ERROR, label="error")
    unrelated = UndirectedNode(content="A different previous error.", label="error")

    result = _strategy([unrelated, matched]).analyze_error(f"{ROWS_ERROR}\n{ROWS_ERROR}", feedback_type="value")

    assert len(result) == 1
    assert result[0] is matched
