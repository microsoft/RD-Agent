from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_agent_scenarios() -> None:
    response = client.get("/api/v1/agent/scenarios")
    assert response.status_code == 200
    payload = response.json()
    assert any(item["name"] == "Finance Data Building" for item in payload)


def test_research_experiments() -> None:
    response = client.get("/api/v1/research/experiments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_receive_endpoint() -> None:
    response = client.post(
        "/receive",
        json={"id": "/tmp/trace", "msg": {"tag": "test", "timestamp": "2020-01-01T00:00:00", "content": {}}},
    )
    assert response.status_code == 200
