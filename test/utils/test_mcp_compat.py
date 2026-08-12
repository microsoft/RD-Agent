from rdagent.components.agent.mcp_compat import MCPServerHTTP

"""pydantic-ai renamed ``MCPServerStreamableHTTP`` to ``MCPServerHTTP`` in 1.x.

The compat layer must resolve to whichever name the installed pydantic-ai
exposes so agent components keep working across 1.x releases.
"""


def test_importable() -> None:
    assert callable(MCPServerHTTP)


def test_construct_with_url() -> None:
    server = MCPServerHTTP("http://127.0.0.1:8080", timeout=5)
    assert str(server.url) == "http://127.0.0.1:8080"
