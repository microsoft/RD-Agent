"""Regression tests for ``CoSTEERRAGStrategyV2.generate_knowledge`` cursor logic.

The cursor (``current_generated_trace_count``) is an attribute on the strategy
instance, but it is interpreted as a position into whatever ``evolving_trace``
is passed in.  When ``CoSTEER.develop()`` is called more than once with the
same strategy instance the cursor must be re-bound to the new trace, otherwise
fresh repair feedback can be silently dropped (issue #1398).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pytest


class _StepProbe:
    """Minimal ``EvoStep`` stand-in that records visits.

    The real ``CoSTEERRAGStrategyV2.generate_knowledge`` body reads
    ``evo_step.evolvable_subjects`` and ``evo_step.feedback`` for each
    in-range trace index.  Empty ``sub_tasks`` / ``sub_workspace_list``
    keep the inner loop a no-op, so we do not need any LLM plumbing to
    exercise the cursor logic.
    """

    def __init__(self) -> None:
        self.visits = 0
        self._es = SimpleNamespace(sub_tasks=[], sub_workspace_list=[])
        self.feedback: list = []

    @property
    def evolvable_subjects(self) -> SimpleNamespace:
        self.visits += 1
        return self._es


def _make_strategy():
    """Build a ``CoSTEERRAGStrategyV2`` without touching the LLM/knowledge-base init.

    ``CoSTEERRAGStrategy.__init__`` reads constructor kwargs that pull a
    knowledge base off disk; for this regression we only care about the
    cursor logic, so we bypass ``__init__`` entirely.
    """

    from rdagent.components.coder.CoSTEER.knowledge_management import (
        CoSTEERRAGStrategyV2,
    )

    strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
    strategy.current_generated_trace_count = 0
    strategy._generated_trace_identity = None
    return strategy


@pytest.mark.offline
class CoSTEERRAGStrategyV2CursorTest(unittest.TestCase):
    def test_fresh_trace_with_same_length_is_ingested(self):
        """A new trace object reusing a stale cursor length must NOT be skipped."""

        strategy = _make_strategy()
        trace1 = [_StepProbe(), _StepProbe()]
        strategy.generate_knowledge(trace1)
        self.assertEqual(strategy.current_generated_trace_count, 2)
        self.assertEqual([s.visits for s in trace1], [1, 1])

        trace2 = [_StepProbe(), _StepProbe()]
        strategy.generate_knowledge(trace2)
        self.assertEqual(
            [s.visits for s in trace2],
            [1, 1],
            "fresh trace2 of stale-cursor length must still be ingested",
        )
        self.assertEqual(strategy.current_generated_trace_count, 2)
        self.assertEqual(strategy._generated_trace_identity, id(trace2))

    def test_extending_same_trace_only_processes_new_steps(self):
        strategy = _make_strategy()
        trace = [_StepProbe()]
        strategy.generate_knowledge(trace)
        self.assertEqual(trace[0].visits, 1)
        self.assertEqual(strategy.current_generated_trace_count, 1)

        new_step = _StepProbe()
        trace.append(new_step)
        strategy.generate_knowledge(trace)
        self.assertEqual(
            trace[0].visits,
            1,
            "step already ingested in a prior call must not be re-processed",
        )
        self.assertEqual(new_step.visits, 1)
        self.assertEqual(strategy.current_generated_trace_count, 2)

    def test_truncated_trace_resets_cursor(self):
        strategy = _make_strategy()
        trace = [_StepProbe(), _StepProbe(), _StepProbe()]
        strategy.generate_knowledge(trace)
        self.assertEqual(strategy.current_generated_trace_count, 3)

        del trace[2]
        del trace[1]
        strategy.generate_knowledge(trace)
        self.assertEqual(
            trace[0].visits,
            2,
            "after truncation the cursor must reset and ingest again",
        )
        self.assertEqual(strategy.current_generated_trace_count, 1)

    def test_idempotent_call_on_same_trace_returns_none(self):
        strategy = _make_strategy()
        trace = [_StepProbe()]
        strategy.generate_knowledge(trace)
        result = strategy.generate_knowledge(trace)
        self.assertIsNone(result)
        self.assertEqual(trace[0].visits, 1)
        self.assertEqual(strategy.current_generated_trace_count, 1)


if __name__ == "__main__":
    unittest.main()
