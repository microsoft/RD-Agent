from http import HTTPStatus

import pytest
from flask.testing import FlaskClient
from rdagent.log.server import debug_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FlaskClient:
    monkeypatch.setattr(debug_app, "msgs_for_frontend", debug_app.msgs_for_frontend.__class__(list))
    return debug_app.app.test_client()


@pytest.mark.offline
def test_receive_accepts_single_dict_payload(client: FlaskClient) -> None:
    response = client.post(
        "/receive",
        json={"id": "trace-1", "msg": {"tag": "A", "content": {"value": 1}}},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.get_json() == {"status": "success"}
    assert debug_app.msgs_for_frontend["trace-1"] == [{"tag": "A", "content": {"value": 1}}]


@pytest.mark.offline
def test_receive_accepts_list_payload_and_preserves_order(client: FlaskClient) -> None:
    response = client.post(
        "/receive",
        json=[
            {"id": "trace-1", "msg": {"tag": "A", "content": {"value": 1}}},
            {"id": "trace-1", "msg": {"tag": "B", "content": {"value": 2}}},
        ],
    )

    assert response.status_code == HTTPStatus.OK
    assert response.get_json() == {"status": "success"}
    assert debug_app.msgs_for_frontend["trace-1"] == [
        {"tag": "A", "content": {"value": 1}},
        {"tag": "B", "content": {"value": 2}},
    ]


@pytest.mark.offline
def test_receive_list_payload_routes_each_entry_to_its_own_id(client: FlaskClient) -> None:
    response = client.post(
        "/receive",
        json=[
            {"id": "trace-1", "msg": {"tag": "A"}},
            {"id": "trace-2", "msg": {"tag": "B"}},
        ],
    )

    assert response.status_code == HTTPStatus.OK
    assert debug_app.msgs_for_frontend["trace-1"] == [{"tag": "A"}]
    assert debug_app.msgs_for_frontend["trace-2"] == [{"tag": "B"}]


@pytest.mark.offline
def test_receive_missing_json_returns_400_not_500(client: FlaskClient) -> None:
    response = client.post("/receive", data="", content_type="application/json")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.get_json() == {"error": "No JSON data received"}
    assert debug_app.msgs_for_frontend == {}


@pytest.mark.offline
@pytest.mark.parametrize(
    "payload",
    [
        {"msg": {"tag": "A"}},  # missing id
        {"id": "trace-1"},  # missing msg
        {"id": "trace-1", "msg": "not-an-object"},  # msg is not an object
        {"id": None, "msg": {"tag": "A"}},  # null id
        "just-a-string",  # not an object at all
        [{"id": "trace-1", "msg": {"tag": "A"}}, {"id": "trace-2"}],  # one malformed entry in a list
    ],
)
def test_receive_rejects_malformed_payloads_without_mutating_state(client: FlaskClient, payload: object) -> None:
    response = client.post("/receive", json=payload)

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "error" in response.get_json()
    # No entry should be recorded, even for the leading well-formed item of a
    # list payload where a later item is malformed.
    assert debug_app.msgs_for_frontend == {}
