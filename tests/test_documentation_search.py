"""Exercise the production feedback caller with a scripted model and HTTP MCP server."""

import asyncio
import json

import anyio
import httpx
import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from rdagent.app.data_science.conf import DataScienceBasePropSetting
from rdagent.components.agent import base, documentation
from rdagent.components.agent.context7.conf import Settings as Context7Settings
from rdagent.components.agent.documentation.conf import Settings

pytestmark = pytest.mark.offline

DOC_URL = "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html"
REFERENCE = f"DataFrame.drop uses the columns parameter. Source: {DOC_URL}"
ERROR = "TypeError: DataFrame.drop() got an unexpected keyword argument 'column'"


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in (
        "CONTEXT7_URL",
        "CONTEXT7_TIMEOUT",
        "CONTEXT7_ENABLE_CACHE",
        "DOCUMENTATION_SEARCH_PROVIDER",
        "DOCUMENTATION_SEARCH_TIMEOUT",
        "DS_ENABLE_MCP_DOCUMENTATION_SEARCH",
        "OPENAI_API_KEY",
        "PARALLEL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(documentation, "SETTINGS", Settings())
    monkeypatch.setattr(documentation, "CONTEXT7_SETTINGS", Context7Settings())
    # Only model inference is scripted. The production resolver, agent, MCP client,
    # JSON-RPC transport and feedback consumer all execute normally.
    monkeypatch.setattr(base, "get_agent_model", lambda: FunctionModel(documentation_model))


async def documentation_model(messages, info):
    assert {tool.name for tool in info.function_tools} == {"web_search", "web_fetch"}
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if not returns:
        return ModelResponse(
            [
                ToolCallPart(
                    "web_search",
                    {
                        "objective": "Find official pandas DataFrame.drop API documentation",
                        "search_queries": ["pandas drop columns parameter"],
                    },
                    tool_call_id="search",
                )
            ]
        )
    assert DOC_URL in json.dumps(returns[0].content)
    if len(returns) == 1:
        return ModelResponse(
            [
                ToolCallPart(
                    "web_fetch", {"urls": [DOC_URL], "objective": "Read the drop signature"}, tool_call_id="fetch"
                )
            ]
        )
    assert "columns" in json.dumps(returns[-1].content)
    return ModelResponse([TextPart(REFERENCE)])


class MCPFixture:
    def __init__(self, failure=None):
        self.requests = []
        self.failure = failure

    async def handle(self, request):
        self.requests.append(request)
        assert str(request.url) == documentation.PARALLEL_MCP_URL
        assert request.headers["user-agent"] == documentation.USER_AGENT
        assert "authorization" not in request.headers
        if request.method != "POST":
            return httpx.Response(405)
        body = json.loads(request.content)
        method = body["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            assert "sampling" not in body["params"]["capabilities"]
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "documentation-test", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "web_search",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "objective": {"type": "string"},
                                "search_queries": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["objective", "search_queries"],
                        },
                    },
                    {
                        "name": "web_fetch",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "urls": {"type": "array", "items": {"type": "string"}},
                                "objective": {"type": "string"},
                            },
                            "required": ["urls"],
                        },
                    },
                ]
            }
        elif method == "tools/call":
            if self.failure == "timeout":
                raise httpx.ReadTimeout("lookup timed out", request=request)
            if self.failure == "http":
                return httpx.Response(503)
            if self.failure == "tool":
                result = {"content": [{"type": "text", "text": "lookup failed"}], "isError": True}
            else:
                params = body["params"]
                if params["name"] == "web_search":
                    assert params["arguments"]["search_queries"] == ["pandas drop columns parameter"]
                    payload = {"results": [{"url": DOC_URL, "excerpts": ["DataFrame.drop columns parameter"]}]}
                else:
                    assert params["name"] == "web_fetch"
                    assert params["arguments"]["urls"] == [DOC_URL]
                    payload = {
                        "results": [{"url": DOC_URL, "excerpts": ["drop(labels=None, *, axis=0, columns=None)"]}]
                    }
                result = {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def install(self, monkeypatch):
        client = httpx.AsyncClient

        def create_client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(self.handle)
            return client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", create_client)


def test_fresh_no_key_search_fetch_and_feedback(monkeypatch):
    fixture = MCPFixture()
    fixture.install(monkeypatch)
    assert not DataScienceBasePropSetting().enable_mcp_documentation_search
    assert isinstance(documentation.get_documentation_agent(), documentation.ParallelDocumentationAgent)
    feedback = documentation.add_documentation_reference(ERROR, enabled=True, required=True)
    assert feedback.startswith(ERROR + "\n\n### API Documentation Reference:")
    assert REFERENCE in feedback
    calls = [json.loads(request.content) for request in fixture.requests if request.method == "POST"]
    assert [call["params"]["name"] for call in calls if call["method"] == "tools/call"] == ["web_search", "web_fetch"]
    assert any(call["method"] == "tools/list" for call in calls)


@pytest.mark.parametrize(
    "enabled,required,error",
    [(False, True, ERROR), (True, False, ERROR), (True, None, ERROR), (True, True, None), (True, True, "")],
)
def test_disabled_or_unnecessary_lookup_makes_no_agent(monkeypatch, enabled, required, error):
    def unexpected_agent():
        pytest.fail("disabled documentation lookup created an agent")

    monkeypatch.setattr(documentation, "get_documentation_agent", unexpected_agent)
    assert documentation.add_documentation_reference(error, enabled=enabled, required=required) == error


@pytest.mark.parametrize(
    "values",
    [
        {"url": "http://localhost:9999/mcp"},
        {"timeout": 13},
        {"enable_cache": True},
        {"url": "http://localhost:8124/mcp"},
    ],
)
def test_existing_context7_environment_settings_preserved(monkeypatch, values):
    for name, value in values.items():
        monkeypatch.setenv(f"CONTEXT7_{name.upper()}", str(value))
    settings = Context7Settings()
    monkeypatch.setattr(documentation, "CONTEXT7_SETTINGS", settings)
    # Context7's own module must see the same real settings instance.
    monkeypatch.setattr("rdagent.components.agent.context7.SETTINGS", settings)
    agent = documentation.get_documentation_agent()
    assert isinstance(agent, documentation.Context7Agent)
    assert agent.enable_cache == settings.enable_cache
    server = agent.agent._user_toolsets[0]
    assert server.url == settings.url
    assert server.timeout == settings.timeout


def test_environment_provider_override_and_timeout(monkeypatch):
    monkeypatch.setenv("DOCUMENTATION_SEARCH_PROVIDER", "parallel")
    monkeypatch.setenv("DOCUMENTATION_SEARCH_TIMEOUT", "17")
    monkeypatch.setattr(documentation, "SETTINGS", Settings())
    monkeypatch.setattr(documentation, "CONTEXT7_SETTINGS", Context7Settings(url="http://localhost:9999/mcp"))
    agent = documentation.get_documentation_agent()
    assert isinstance(agent, documentation.ParallelDocumentationAgent)
    server = agent.agent._user_toolsets[0]
    assert server.timeout == server.read_timeout == 17
    monkeypatch.setattr(documentation, "SETTINGS", Settings(provider="context7"))
    assert isinstance(documentation.get_documentation_agent(), documentation.Context7Agent)


@pytest.mark.parametrize("values", [{"provider": "missing"}, {"timeout": 0}, {"timeout": -1}])
def test_invalid_settings_fail(values):
    with pytest.raises(ValidationError):
        Settings(**values)


@pytest.mark.parametrize("failure", ["timeout", "http", "tool"])
def test_transport_and_tool_failure_preserve_error_without_fallback(monkeypatch, failure):
    fixture = MCPFixture(failure)
    fixture.install(monkeypatch)
    assert documentation.add_documentation_reference(ERROR, enabled=True, required=True) == ERROR
    assert any(
        json.loads(request.content).get("method") == "tools/call"
        for request in fixture.requests
        if request.method == "POST"
    )


def test_context7_failure_does_not_try_parallel(monkeypatch):
    monkeypatch.setattr(documentation, "SETTINGS", Settings(provider="context7"))

    class FailingContext7:
        def query(self, query):
            raise TimeoutError("Context7 timed out")

    monkeypatch.setattr(documentation, "Context7Agent", FailingContext7)
    monkeypatch.setattr(documentation, "ParallelDocumentationAgent", lambda: pytest.fail("unexpected fallback"))
    assert documentation.add_documentation_reference(ERROR, enabled=True, required=True) == ERROR


def test_feedback_cancellation_is_not_swallowed(monkeypatch):
    class CancelledAgent:
        def query(self, query):
            raise asyncio.CancelledError

    monkeypatch.setattr(documentation, "get_documentation_agent", CancelledAgent)
    with pytest.raises(asyncio.CancelledError):
        documentation.add_documentation_reference(ERROR, enabled=True, required=True)


def test_empty_lookup_preserves_original_error(monkeypatch):
    async def empty_model(messages, info):
        return ModelResponse([TextPart("")])

    monkeypatch.setattr(base, "get_agent_model", lambda: FunctionModel(empty_model))
    fixture = MCPFixture()
    fixture.install(monkeypatch)
    assert documentation.add_documentation_reference(ERROR, enabled=True, required=True) == ERROR


def test_cancellation_propagates_and_closes_transport(monkeypatch):
    fixture = MCPFixture()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    normal_handler = fixture.handle

    async def blocking_handler(request):
        body = json.loads(request.content) if request.method == "POST" else {}
        if body.get("method") == "tools/call":
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        return await normal_handler(request)

    fixture.handle = blocking_handler
    fixture.install(monkeypatch)

    async def run():
        agent = documentation.get_documentation_agent()
        async with agent.agent:
            async with anyio.create_task_group() as group:
                group.start_soon(agent.agent.run, ERROR)
                with anyio.fail_after(5):
                    await started.wait()
                group.cancel_scope.cancel()
        await asyncio.wait_for(cancelled.wait(), timeout=5)
        assert not agent.agent._user_toolsets[0].is_running

    asyncio.run(run())
