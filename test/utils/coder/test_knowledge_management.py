"""Tests for CoSTEER knowledge_management module."""

import unittest
from types import SimpleNamespace

from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERRAGStrategyV2,
)
from rdagent.components.knowledge_management.graph import (
    UndirectedNode,
)


class GraphStub:
    """Minimal graph stub for testing analyze_error."""

    def __init__(self, nodes):
        self.nodes = nodes

    def get_all_nodes_by_label_list(self, labels):
        assert labels == ["error"]
        return self.nodes


class TestAnalyzeError(unittest.TestCase):
    """Tests for CoSTEERRAGStrategyV2.analyze_error."""

    def test_no_existing_error_nodes(self):
        """When graph has no error nodes, all error contents are returned as strings."""
        strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
        strategy.knowledgebase = SimpleNamespace(graph=GraphStub([]))

        feedback = "The source dataframe and the ground truth dataframe have different rows count."

        result = strategy.analyze_error(feedback, feedback_type="value")

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)
        self.assertEqual(result[0], feedback)

    def test_parsed_error_with_matching_existing_node(self):
        """When a parsed error matches an existing graph node, the node is returned."""
        matched_content = "The source dataframe and the ground truth dataframe have different rows count."

        matched_node = UndirectedNode(
            content=matched_content,
            label="error",
        )

        strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
        strategy.knowledgebase = SimpleNamespace(graph=GraphStub([matched_node]))

        result = strategy.analyze_error(matched_content, feedback_type="value")

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], UndirectedNode)
        self.assertEqual(result[0].content, matched_content)

    def test_parsed_error_with_only_nonmatching_existing_nodes(self):
        """When a parsed error has no matching node, the string is returned."""
        unrelated_content = "A different previous error."
        missing_content = "The source dataframe and the ground truth dataframe have different rows count."

        unrelated_node = UndirectedNode(
            content=unrelated_content,
            label="error",
        )

        strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
        strategy.knowledgebase = SimpleNamespace(graph=GraphStub([unrelated_node]))

        result = strategy.analyze_error(missing_content, feedback_type="value")

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)
        self.assertEqual(result[0], missing_content)

    def test_multiple_parsed_errors_with_one_match_and_one_nonmatch(self):
        """Multiple parsed errors: one matches, one doesn't."""
        matched_content = "The source dataframe and the ground truth dataframe have different rows count."
        missing_content = "Some values differ by more than the tolerance of 1e-6."

        matched_node = UndirectedNode(
            content=matched_content,
            label="error",
        )

        strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
        strategy.knowledgebase = SimpleNamespace(graph=GraphStub([matched_node]))

        feedback = matched_content + "\n" + missing_content
        result = strategy.analyze_error(feedback, feedback_type="value")

        self.assertEqual(len(result), 2)
        # First should be the matched node
        self.assertIsInstance(result[0], UndirectedNode)
        self.assertEqual(result[0].content, matched_content)
        # Second should be the string
        self.assertIsInstance(result[1], str)
        self.assertEqual(result[1], missing_content)

    def test_unrelated_node_before_matching_node(self):
        """Graph nodes ordered with unrelated node before matching node."""
        matched_content = "The source dataframe and the ground truth dataframe have different rows count."
        unrelated_content = "A different previous error."

        matched_node = UndirectedNode(content=matched_content, label="error")
        unrelated_node = UndirectedNode(content=unrelated_content, label="error")

        strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
        strategy.knowledgebase = SimpleNamespace(
            graph=GraphStub([unrelated_node, matched_node])  # unrelated first
        )

        result = strategy.analyze_error(matched_content, feedback_type="value")

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], UndirectedNode)
        self.assertEqual(result[0].content, matched_content)

    def test_matching_node_before_unrelated_node(self):
        """Graph nodes ordered with matching node before unrelated node."""
        matched_content = "The source dataframe and the ground truth dataframe have different rows count."
        unrelated_content = "A different previous error."

        matched_node = UndirectedNode(content=matched_content, label="error")
        unrelated_node = UndirectedNode(content=unrelated_content, label="error")

        strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
        strategy.knowledgebase = SimpleNamespace(
            graph=GraphStub([matched_node, unrelated_node])  # matching first
        )

        result = strategy.analyze_error(matched_content, feedback_type="value")

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], UndirectedNode)
        self.assertEqual(result[0].content, matched_content)

    def test_preserves_parsed_error_content_order(self):
        """The order of parsed error contents in the result matches the input order."""
        content1 = "The source dataframe and the ground truth dataframe have different rows count."
        content2 = "Some values differ by more than the tolerance of 1e-6."
        content3 = "The source dataframe and the ground truth dataframe have different index."

        node1 = UndirectedNode(content=content1, label="error")
        node3 = UndirectedNode(content=content3, label="error")

        strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
        strategy.knowledgebase = SimpleNamespace(graph=GraphStub([node1, node3]))

        feedback = content1 + "\n" + content2 + "\n" + content3
        result = strategy.analyze_error(feedback, feedback_type="value")

        self.assertEqual(len(result), 3)
        # First: matched node for content1
        self.assertIsInstance(result[0], UndirectedNode)
        self.assertEqual(result[0].content, content1)
        # Second: string for content2 (no match)
        self.assertIsInstance(result[1], str)
        self.assertEqual(result[1], content2)
        # Third: matched node for content3
        self.assertIsInstance(result[2], UndirectedNode)
        self.assertEqual(result[2].content, content3)

    def test_no_duplicate_string_node_pair_for_same_error_content(self):
        """The original bug: a matched error should not produce both string and node."""
        matched_content = "The source dataframe and the ground truth dataframe have different rows count."
        missing_content = "Some values differ by more than the tolerance of 1e-6."

        unrelated_node = UndirectedNode(
            content="A different previous error.",
            label="error",
        )
        matched_node = UndirectedNode(
            content=matched_content,
            label="error",
        )

        strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
        strategy.knowledgebase = SimpleNamespace(graph=GraphStub([unrelated_node, matched_node]))

        feedback = matched_content + "\n" + missing_content
        result = strategy.analyze_error(feedback, feedback_type="value")

        # Should have exactly 2 items: matched_node and missing_content string
        self.assertEqual(len(result), 2)

        # Check no duplicate for the matched content
        matched_count = sum(
            1 for item in result if isinstance(item, UndirectedNode) and item.content == matched_content
        ) + sum(1 for item in result if isinstance(item, str) and item == matched_content)
        self.assertEqual(matched_count, 1, "Matched content should appear exactly once")

        # First item should be the node, not the string
        self.assertIsInstance(result[0], UndirectedNode)
        self.assertEqual(result[0].content, matched_content)
        # Second item should be the missing content string
        self.assertIsInstance(result[1], str)
        self.assertEqual(result[1], missing_content)


if __name__ == "__main__":
    unittest.main()
