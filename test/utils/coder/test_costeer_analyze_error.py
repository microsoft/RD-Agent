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


@pytest.mark.offline
def test_analyze_error_returns_matched_node_once() -> None:
    other = UndirectedNode(content="A different previous error.", label="error")
    matched = UndirectedNode(content=ROWS_ERROR, label="error")
    graph = UndirectedGraph()
    graph.nodes = {node.id: node for node in (other, matched)}
    strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
    strategy.knowledgebase = SimpleNamespace(graph=graph)
    assert strategy.analyze_error(ROWS_ERROR, feedback_type="value") == [matched]
