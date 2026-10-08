from collections import Counter
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.evolving_strategy import MultiProcessEvolvingStrategy
from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERRAGStrategyV2
from rdagent.core.experiment import Experiment, Task


class CountingStrategy(MultiProcessEvolvingStrategy):
    """Implement every scheduled task and count how often each task is implemented."""

    def __init__(self, settings: CoSTEERSettings) -> None:
        super().__init__(scen=None, settings=settings)
        self.calls: Counter = Counter()

    def implement_one_task(self, target_task: Task, *args) -> dict[str, str]:
        self.calls[target_task.name] += 1
        return {"factor.py": str(self.calls[target_task.name])}

    def assign_code_list_to_evo(self, code_list: list[dict], evo) -> None:
        # The base class implementation returns `evo`; the evolving strategy ignores that
        # return value, so assigning the workspaces in place is equivalent here.
        for index, code in enumerate(code_list):
            workspace = Mock(file_dict=code)
            workspace.copy.side_effect = lambda code=code: Mock(file_dict=deepcopy(code))
            evo.sub_workspace_list[index] = workspace


class DeterministicEvaluator:
    """Accept a task once it has been implemented ``required_attempts`` times.

    Mirrors the generator contract of ``CoSTEERMultiEvaluator.evaluate_iter``: the first
    ``next()`` consumes the kick-off yield, then each ``send`` receives an evolving item
    and gets one feedback back, and the iteration ends with ``StopIteration`` carrying the
    overall feedback.
    """

    def __init__(self, required_attempts: dict[str, int]) -> None:
        self.required_attempts = required_attempts

    def evaluate_iter(self, **kwargs):
        evo = yield CoSTEERMultiFeedback([])
        feedback = CoSTEERMultiFeedback(
            [
                CoSTEERSingleFeedback(
                    execution="local test",
                    return_checking=None,
                    code="local test",
                    final_decision=int(workspace.file_dict["factor.py"]) >= self.required_attempts.get(task.name, 1),
                )
                for task, workspace in zip(evo.sub_tasks, evo.sub_workspace_list)
            ]
        )
        yield feedback
        return feedback


@pytest.fixture
def make_coder(monkeypatch):
    """Build a coder with the real develop/evolution/knowledge logic and mocked I/O."""

    costeer_module = import_module("rdagent.components.coder.CoSTEER")
    strategy_module = import_module("rdagent.components.coder.CoSTEER.evolving_strategy")
    agent_module = import_module("rdagent.core.evolving_agent")

    monkeypatch.setattr(costeer_module, "logger", Mock())
    monkeypatch.setattr(agent_module, "logger", MagicMock())
    monkeypatch.setattr(strategy_module, "multiprocessing_wrapper", lambda calls, n: [f(*args) for f, args in calls])
    monkeypatch.setattr(costeer_module, "RD_Agent_TIMER_wrapper", SimpleNamespace(timer=SimpleNamespace(started=False)))

    def _factory(required_attempts=None, knowledge_self_gen=True):
        settings = CoSTEERSettings(max_loop=5, enable_filelock=False)
        rag = CoSTEERRAGStrategyV2.__new__(CoSTEERRAGStrategyV2)
        rag.settings = settings
        rag.dump_knowledge_base_path = None
        rag.current_generated_trace_count = 0
        rag.knowledgebase = SimpleNamespace(
            success_task_to_knowledge_dict={},
            task_to_component_nodes={},
            working_trace_knowledge={},
            working_trace_error_analysis={},
            update_success_task=Mock(),
        )
        rag.analyze_component = Mock(return_value=[])
        rag.analyze_error = Mock(return_value=[])
        rag.load_dumped_knowledge_base = Mock()
        rag.dump_knowledge_base = Mock()
        for method in ("former_trace_query", "component_query", "error_query"):
            setattr(rag, method, Mock(side_effect=lambda evo, knowledge, *args, **kwargs: knowledge))

        monkeypatch.setattr(costeer_module, "CoSTEERRAGStrategyV2", lambda **kwargs: rag)

        strategy = CountingStrategy(settings)
        coder = CoSTEER(
            scen=None,
            settings=settings,
            eva=DeterministicEvaluator(required_attempts or {}),
            es=strategy,
            evolving_version=2,
            knowledge_self_gen=knowledge_self_gen,
        )
        return coder, strategy

    return _factory


@pytest.mark.offline
@pytest.mark.parametrize("previous_rounds", [1, 3])
def test_new_develop_preserves_knowledge_and_processes_first_round(make_coder, previous_rounds) -> None:
    """A new ``develop`` call must record its own trace instead of skipping it.

    ``current_generated_trace_count`` used to keep the value left over from the previous
    ``develop`` call while the new ``RAGEvoAgent`` started with an empty evolving trace.
    The guard in ``generate_knowledge`` therefore returned immediately, and none of the
    tasks implemented by the new call reached the knowledge base.
    """
    coder, strategy = make_coder({"previous": previous_rounds, "retry": 2})
    previous = Task("previous")
    coder.develop(Experiment([previous]))

    rag = coder.rag
    assert rag.current_generated_trace_count == previous_rounds
    previous_knowledge = rag.knowledgebase.success_task_to_knowledge_dict[previous.get_task_information()]

    # the second call carries the already known task plus two new ones
    success, retry = Task("success"), Task("retry")
    coder.develop(Experiment([previous, success, retry]))

    assert coder.rag is rag
    # the known task is reused from the query and is not implemented again
    assert strategy.calls == {"previous": previous_rounds, "success": 1, "retry": 2}
    assert len(coder.evolve_agent.evolving_trace) == 2
    assert rag.current_generated_trace_count == 2
    # the knowledge base is preserved across calls
    assert rag.knowledgebase.success_task_to_knowledge_dict[previous.get_task_information()] is previous_knowledge
    assert len(rag.knowledgebase.working_trace_knowledge[success.get_task_information()]) == 1
    retry_history = rag.knowledgebase.working_trace_knowledge[retry.get_task_information()]
    assert [record.feedback.final_decision for record in retry_history] == [False, True]

    # knowledge is only generated once per recorded trace entry
    before = {key: len(value) for key, value in rag.knowledgebase.working_trace_knowledge.items()}
    updates = rag.knowledgebase.update_success_task.call_count
    rag.generate_knowledge(coder.evolve_agent.evolving_trace)
    assert {key: len(value) for key, value in rag.knowledgebase.working_trace_knowledge.items()} == before
    assert rag.knowledgebase.update_success_task.call_count == updates


@pytest.mark.offline
def test_develop_with_knowledge_self_generation_disabled(make_coder) -> None:
    coder, strategy = make_coder(knowledge_self_gen=False)
    coder.rag.current_generated_trace_count = 3

    for _ in range(2):
        coder.develop(Experiment([Task("success")]))

    assert strategy.calls == {"success": 2}
    assert coder.rag.knowledgebase.success_task_to_knowledge_dict == {}
    assert coder.rag.knowledgebase.working_trace_knowledge == {}
    coder.rag.analyze_component.assert_not_called()
    coder.rag.dump_knowledge_base.assert_not_called()
