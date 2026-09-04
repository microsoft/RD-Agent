import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_health_check_module():
    logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    docker_module = ModuleType("docker")
    docker_module.from_env = lambda: None
    docker_module.errors = SimpleNamespace(DockerException=Exception)

    fire_module = ModuleType("fire")
    fire_module.Fire = lambda *args, **kwargs: None

    litellm_module = ModuleType("litellm")
    litellm_module.completion = lambda *args, **kwargs: None
    litellm_module.embedding = lambda *args, **kwargs: None

    litellm_utils_module = ModuleType("litellm.utils")
    litellm_utils_module.ModelResponse = object

    rdagent_log_module = ModuleType("rdagent.log")
    rdagent_log_module.rdagent_logger = logger

    rdagent_utils_env_module = ModuleType("rdagent.utils.env")
    rdagent_utils_env_module.cleanup_container = lambda *args, **kwargs: None

    stubs = {
        "docker": docker_module,
        "fire": fire_module,
        "litellm": litellm_module,
        "litellm.utils": litellm_utils_module,
        "rdagent.log": rdagent_log_module,
        "rdagent.utils.env": rdagent_utils_env_module,
    }

    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module_path = Path(__file__).resolve().parents[2] / "rdagent" / "app" / "utils" / "health_check.py"
        spec = importlib.util.spec_from_file_location("test_health_check_module", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


health_check = _load_health_check_module()


def test_env_check_returns_without_credentials(monkeypatch):
    calls = []

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(health_check, "test_chat", lambda *args, **kwargs: calls.append("chat"))
    monkeypatch.setattr(health_check, "test_embedding", lambda *args, **kwargs: calls.append("embedding"))

    health_check.env_check()

    assert calls == []


def test_env_check_returns_with_empty_credentials(monkeypatch):
    calls = []

    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("CHAT_MODEL", "test-chat")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setattr(
        health_check,
        "test_chat",
        lambda *args, **kwargs: calls.append("chat"),
    )
    monkeypatch.setattr(
        health_check,
        "test_embedding",
        lambda *args, **kwargs: calls.append("embedding"),
    )

    health_check.env_check()

    assert calls == []


def test_env_check_returns_when_required_models_missing(monkeypatch):
    calls = []

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(health_check, "test_chat", lambda *args, **kwargs: calls.append("chat"))
    monkeypatch.setattr(health_check, "test_embedding", lambda *args, **kwargs: calls.append("embedding"))

    health_check.env_check()

    assert calls == []


def test_check_and_list_free_ports_uses_requested_start_port(monkeypatch):
    messages = []

    monkeypatch.setattr(
        health_check,
        "is_port_in_use",
        lambda port: port == 21000,
    )
    monkeypatch.setattr(health_check.logger, "warning", lambda message: messages.append(message))

    health_check.check_and_list_free_ports(start_port=21000, max_ports=3)

    assert messages
    assert "21000" in messages[0]
