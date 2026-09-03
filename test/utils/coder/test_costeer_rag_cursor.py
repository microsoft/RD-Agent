"""Regression tests for issue #1398.

``CoSTEERRAGStrategyV2`` keeps a persistent ``current_generated_trace_count``
cursor.  When a fresh ``evolving_trace`` object is supplied with the same
length as a previously processed trace, the cursor was previously interpreted
as already pointing at the end of the new trace and ``generate_knowledge``
returned early -- silently dropping the latest repair feedback and causing
already-successful candidates to be rescheduled in subsequent CoSTEER repair
rounds.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

import pytest


class _FakeTask:
    def __init__(self, info: str) -> None:
        self._info = info

    def get_task_information(self) -> str:
        return self._info


class _FakeWorkspace:
    def copy(self) -> "_FakeWorkspace":
        return _FakeWorkspace()

    @property
    def all_codes(self) -> str:
        return ""


@dataclass
class _FakeFeedback:
    final_decision: bool = True
    return_checking: str | None = None
    execution: str | None = None

    def __str__(self) -> str:  # mirrors ``CoSTEERSingleFeedback.__str__`` usage
        return f"final_decision={self.final_decision}"


@dataclass
class _FakeSubjects:
    sub_tasks: list[_FakeTask]
    sub_workspace_list: list[_FakeWorkspace]


@dataclass
class _FakeEvoStep:
    evolvable_subjects: _FakeSubjects
    feedback: list[_FakeFeedback]
    queried_knowledge: object | None = None


@dataclass
class _FakeKnowledgeBase:
    success_task_to_knowledge_dict: dict = field(default_factory=dict)
    working_trace_knowledge: dict = field(default_factory=dict)
    working_trace_error_analysis: dict = field(default_factory=dict)
    task_to_component_nodes: dict = field(default_factory=dict)
    update_calls: list[str] = field(default_factory=list)

    def update_success_task(self, task_info: str) -> None:
        self.update_calls.append(task_info)


def _make_trace(task_info: str, decision: bool = True) -> _FakeEvoStep:
    task = _FakeTask(task_info)
    return _FakeEvoStep(
        evolvable_subjects=_FakeSubjects(
            sub_tasks=[task],
            sub_workspace_list=[_FakeWorkspace()],
        ),
        feedback=[_FakeFeedback(final_decision=decision)],
    )


def _build_strategy():
    """Construct a ``CoSTEERRAGStrategyV2`` whose heavy collaborators are
    replaced with stubs so we can drive ``generate_knowledge`` in isolation."""

    from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
    from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERRAGStrategyV2

    strategy = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
    strategy.current_generated_trace_count = 0
    strategy._generated_trace_identity = None
    strategy.settings = CoSTEERSettings()
    strategy.knowledgebase = _FakeKnowledgeBase()
    strategy.dump_knowledge_base_path = None
    strategy.analyze_component = lambda _info: []  # type: ignore[method-assign]
    strategy.analyze_error = lambda _msg, feedback_type="execution": []  # type: ignore[method-assign]
    return strategy


@pytest.mark.offline
class CoSTEERRAGStrategyV2CursorTest(unittest.TestCase):
    """Issue #1398: cursor must be rebound when a fresh ``evolving_trace``
    object is supplied."""

    def test_fresh_trace_with_same_length_is_still_ingested(self) -> None:
        strategy = _build_strategy()

        first_trace = [_make_trace("task_alpha")]
        strategy.generate_knowledge(first_trace)
        self.assertEqual(strategy.current_generated_trace_count, 1)
        self.assertIn("task_alpha", strategy.knowledgebase.success_task_to_knowledge_dict)

        # New ``develop()`` run supplies a fresh trace object with the same
        # length and a different task.  The cursor must reset so the new
        # task's feedback is ingested.
        second_trace = [_make_trace("task_beta")]
        self.assertEqual(len(second_trace), strategy.current_generated_trace_count)
        self.assertIsNot(second_trace, first_trace)

        strategy.generate_knowledge(second_trace)

        self.assertIn(
            "task_beta",
            strategy.knowledgebase.success_task_to_knowledge_dict,
            msg="Fresh trace of equal length must not be silently skipped (issue #1398).",
        )
        self.assertEqual(strategy.current_generated_trace_count, 1)

    def test_same_trace_object_is_not_reprocessed(self) -> None:
        strategy = _build_strategy()

        trace = [_make_trace("task_alpha"), _make_trace("task_beta")]
        strategy.generate_knowledge(trace)
        update_count_after_first = len(strategy.knowledgebase.update_calls)

        # Calling again with the same object must short-circuit -- otherwise
        # we would record the same knowledge twice.
        strategy.generate_knowledge(trace)
        self.assertEqual(
            len(strategy.knowledgebase.update_calls),
            update_count_after_first,
            msg="Same trace object must not be re-ingested.",
        )

    def test_cursor_resets_when_trace_truncates_below_cursor(self) -> None:
        strategy = _build_strategy()

        # Simulate a stale cursor that points past the end of the current trace,
        # which can occur when a fresh trace happens to reuse a familiar id but
        # contain fewer steps.
        strategy.current_generated_trace_count = 5
        strategy._generated_trace_identity = None

        trace = [_make_trace("task_gamma")]
        strategy.generate_knowledge(trace)

        self.assertEqual(strategy.current_generated_trace_count, 1)
        self.assertIn("task_gamma", strategy.knowledgebase.success_task_to_knowledge_dict)


if __name__ == "__main__":
    unittest.main()
