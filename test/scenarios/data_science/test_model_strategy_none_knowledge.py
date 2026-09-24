from types import SimpleNamespace

import pytest

import rdagent.components.coder.data_science.model as model_module
from rdagent.components.coder.data_science.model import (
    ModelMultiProcessEvolvingStrategy,
)

pytestmark = pytest.mark.offline


def test_model_strategy_accepts_none_queried_knowledge(monkeypatch):
    class FakeTemplate:
        def r(self, *args, **kwargs):
            return "prompt"

    monkeypatch.setattr(model_module, "T", lambda *args, **kwargs: FakeTemplate())
    monkeypatch.setattr(model_module.DS_RD_SETTING, "spec_enabled", True)
    monkeypatch.setattr(
        model_module.PythonBatchEditOut,
        "extract_output",
        lambda *args, **kwargs: {"model_generated.py": "new code"},
    )

    class FakeBackend:
        def build_messages_and_create_chat_completion(self, **kwargs):
            return "response"

    monkeypatch.setattr(model_module, "APIBackend", FakeBackend)

    strategy = object.__new__(ModelMultiProcessEvolvingStrategy)
    strategy.scen = SimpleNamespace(get_scenario_all_desc=lambda eda_output=None: "scenario")

    target_task = SimpleNamespace(
        name="model_1",
        get_task_information=lambda: "model task",
    )

    class FakeWorkspace:
        file_dict = {
            "model_1.py": "old code",
            "feature.py": "feature code",
            "load_data.py": "loader code",
            "spec/model.md": "model spec",
        }

        def get_codes(self, pattern):
            return {"model_1.py": "old code"}

    result = strategy.implement_one_task(
        target_task=target_task,
        queried_knowledge=None,
        workspace=FakeWorkspace(),
    )

    assert result == {"model_1.py": "new code"}
