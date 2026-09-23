from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from flask.testing import FlaskClient

import rdagent.log.server.app as server
import rdagent.log.ui.storage as web_storage
from rdagent.log.server.security import (
    normalize_origin,
    parse_competition,
    resolve_within,
    validate_scenario,
    validate_upload_filename,
)


class _Response:
    status_code = HTTPStatus.OK
    text = "ok"


@pytest.mark.offline
def test_validate_scenario_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        validate_scenario("../Data Science")


@pytest.mark.offline
def test_parse_competition_accepts_only_mle_bench_slug() -> None:
    assert parse_competition("MLE-Bench:aerial-cactus-identification") == "aerial-cactus-identification"
    for value in (None, "aerial-cactus", "MLE-Bench:../../tmp", "MLE-Bench:a;id"):
        with pytest.raises(ValueError, match=r"Competition|Invalid"):
            parse_competition(value)


@pytest.mark.offline
def test_resolve_within_rejects_escape(tmp_path: Path) -> None:
    assert resolve_within(tmp_path, "scenario", "trace").is_relative_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        resolve_within(tmp_path, "..", "outside")


@pytest.mark.offline
def test_upload_filename_rejects_executable_formats() -> None:
    assert validate_upload_filename("report.pdf") == "report.pdf"
    for filename in ("payload.pkl", "payload.PICKLE", "script.py", ""):
        with pytest.raises(ValueError, match=r"upload|file type"):
            validate_upload_filename(filename)


@pytest.mark.offline
def test_log_server_requires_authentication_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "secret-token")
    client = server.app.test_client()

    assert client.get("/traces").status_code == HTTPStatus.UNAUTHORIZED
    response = client.get("/traces", headers={"Authorization": "Bearer secret-token"})
    assert response.status_code == HTTPStatus.OK


@pytest.mark.offline
@pytest.mark.parametrize(
    ("token", "expected_authorization"),
    [("secret-token", "Bearer secret-token"), ("", None)],
)
def test_web_storage_authenticates_internal_receive_requests(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    expected_authorization: str | None,
) -> None:
    request_headers: dict[str, str] = {}

    def fake_post(url: str, *, json: object, headers: dict[str, str], timeout: int) -> _Response:
        assert url == "http://localhost:19899/receive"
        assert json == {"id": "trace", "msg": {"tag": "test"}}
        assert timeout == 1
        request_headers.update(headers)
        return _Response()

    monkeypatch.setattr(web_storage.UI_SETTING, "server_auth_token", token)
    monkeypatch.setattr(web_storage.requests, "post", fake_post)
    storage = web_storage.WebStorage(port=19899, path="trace")
    monkeypatch.setattr(
        storage,
        "_obj_to_json",
        lambda **_kwargs: {"id": "trace", "msg": {"tag": "test"}},
    )

    assert storage.log("message", "test") == "200 ok"
    assert request_headers.get("Authorization") == expected_authorization


@pytest.mark.offline
def test_upload_rejects_unknown_scenario_before_writing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "secret-token")
    monkeypatch.setattr(server, "upload_folder_path", tmp_path / "uploads")
    client = server.app.test_client()

    response = client.post(
        "/upload",
        data={"scenario": "../Data Science"},
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


@pytest.mark.offline
def test_upload_rejects_pickle_before_starting_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "secret-token")
    monkeypatch.setattr(server, "upload_folder_path", tmp_path / "uploads")
    monkeypatch.setattr(server, "log_folder_path", tmp_path / "traces")
    client = server.app.test_client()

    response = client.post(
        "/upload",
        data={"scenario": "Finance Data Building", "files": (BytesIO(b"payload"), "payload.pkl")},
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert not (tmp_path / "uploads").exists()


@pytest.fixture
def protected_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[FlaskClient, Mock]:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "secret-token")
    monkeypatch.setitem(server.app.config, "CORS_ALLOWED_ORIGINS", {"https://ui.example.com"})
    monkeypatch.setattr(server, "upload_folder_path", tmp_path / "uploads")
    monkeypatch.setattr(server, "log_folder_path", tmp_path / "traces")
    monkeypatch.setattr(server, "rdagent_processes", {})
    task = Mock()
    monkeypatch.setattr(server, "RDAgentTask", task)
    return server.app.test_client(), task


_API_REQUESTS = [
    ("POST", "/upload"),
    ("POST", "/control"),
    ("POST", "/receive"),
    ("POST", "/trace"),
    ("POST", "/user_interaction/submit"),
    ("GET", "/traces"),
    ("GET", "/stdout"),
    ("GET", "/test"),
]


@pytest.mark.offline
@pytest.mark.parametrize(("method", "path"), _API_REQUESTS)
@pytest.mark.parametrize("token", ["", "secret-token"])
def test_all_apis_fail_closed_without_credentials(
    protected_server: tuple[FlaskClient, Mock],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    method: str,
    path: str,
    token: str,
) -> None:
    client, task = protected_server
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", token)
    response = client.open(path, method=method)
    assert response.status_code == (HTTPStatus.UNAUTHORIZED if token else HTTPStatus.SERVICE_UNAVAILABLE)
    task.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.offline
@pytest.mark.parametrize(("method", "path"), _API_REQUESTS)
@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://localhost.evil", "https://uiXexample.com"])
def test_untrusted_origins_cannot_reach_apis(
    protected_server: tuple[FlaskClient, Mock],
    tmp_path: Path,
    method: str,
    path: str,
    origin: str,
) -> None:
    client, task = protected_server
    client.set_cookie("rdagent_auth", "secret-token")
    response = client.open(path, method=method, headers={"Origin": origin})
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert "Access-Control-Allow-Origin" not in response.headers
    task.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.offline
@pytest.mark.parametrize("content_type", ["application/x-www-form-urlencoded", "multipart/form-data", "text/plain"])
def test_cross_site_simple_post_has_no_side_effects(
    protected_server: tuple[FlaskClient, Mock],
    tmp_path: Path,
    content_type: str,
) -> None:
    client, task = protected_server
    response = client.post(
        "/upload",
        data="scenario=General+Model+Implementation&files=http://127.0.0.1/private",
        content_type=content_type,
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == HTTPStatus.FORBIDDEN
    task.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.offline
@pytest.mark.parametrize("headers", [{}, {"Referer": "https://evil.example/path"}, {"Sec-Fetch-Site": "same-site"}])
def test_cookie_post_requires_trusted_provenance(
    protected_server: tuple[FlaskClient, Mock],
    tmp_path: Path,
    headers: dict[str, str],
) -> None:
    client, task = protected_server
    client.set_cookie("rdagent_auth", "secret-token")
    response = client.post("/upload", data={"scenario": "Finance Data Building"}, headers=headers)
    assert response.status_code == HTTPStatus.FORBIDDEN
    task.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.offline
@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://localhost"},
        {"Referer": "http://localhost/playground"},
        {"Origin": "https://ui.example.com"},
        {"Authorization": "Bearer secret-token"},
    ],
)
def test_authenticated_report_upload_starts_task(
    protected_server: tuple[FlaskClient, Mock],
    headers: dict[str, str],
) -> None:
    client, task = protected_server
    if "Authorization" not in headers:
        client.set_cookie("rdagent_auth", "secret-token")
    response = client.post(
        "/upload",
        data={"scenario": "General Model Implementation", "files": (BytesIO(b"%PDF-test"), "report.pdf")},
        headers=headers,
    )
    assert response.status_code == HTTPStatus.OK
    task.return_value.start.assert_called_once()
    report_path = Path(task.call_args.kwargs["kwargs"]["report_file_path"])
    assert report_path.is_relative_to(server.upload_folder_path)
    assert report_path.read_bytes() == b"%PDF-test"


@pytest.mark.offline
@pytest.mark.parametrize(
    "value",
    [
        "http://127.0.0.1/private",
        "http://169.254.169.254/",
        "https://example.com/report.pdf",
        "/etc/passwd",
        "file:///etc/passwd",
        "",
    ],
)
def test_report_text_input_is_rejected_before_side_effects(
    protected_server: tuple[FlaskClient, Mock],
    tmp_path: Path,
    value: str,
) -> None:
    client, task = protected_server
    response = client.post(
        "/upload",
        data={"scenario": "General Model Implementation", "files": value},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    task.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.offline
def test_allowed_preflight_does_not_authorize_actual_request(protected_server: tuple[FlaskClient, Mock]) -> None:
    client, task = protected_server
    headers = {"Origin": "https://ui.example.com", "Access-Control-Request-Method": "POST"}
    response = client.options("/upload", headers=headers)
    assert response.status_code == HTTPStatus.OK
    assert response.headers["Access-Control-Allow-Origin"] == "https://ui.example.com"
    assert client.post("/upload", headers=headers).status_code == HTTPStatus.UNAUTHORIZED
    task.assert_not_called()


@pytest.mark.offline
@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "0.0.0.0"])  # noqa: S104
def test_startup_requires_token_on_every_interface(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "")
    run = Mock()
    monkeypatch.setattr(server.app, "run", run)
    with pytest.raises(ValueError, match="UI_SERVER_AUTH_TOKEN"):
        server.main(host=host)
    run.assert_not_called()


@pytest.mark.offline
def test_browser_session_cookie_and_token_response_headers(protected_server: tuple[FlaskClient, Mock]) -> None:
    client, _ = protected_server
    response = client.get("/?token=secret-token", base_url="https://localhost")
    assert response.status_code == HTTPStatus.FOUND
    cookie = response.headers["Set-Cookie"]
    assert all(flag in cookie for flag in ("HttpOnly", "SameSite=Strict", "Secure"))
    assert response.headers["Location"] == "/"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.offline
@pytest.mark.parametrize(
    "value",
    [
        "null",
        "*",
        "https://*.example.com",
        "https://example.com/path",
        "https://user@example.com",
        "https://example.com:bad",
        "https://a https://b",
    ],
)
def test_invalid_origin_configuration(value: str) -> None:
    assert normalize_origin(value) is None


@pytest.mark.offline
@pytest.mark.parametrize("authorization", ["Bearer wrong-token", "Basic secret-token", "Bearer ", "Bearer 错误"])
def test_invalid_authorization_cannot_fall_back_to_cookie(
    protected_server: tuple[FlaskClient, Mock],
    authorization: str,
) -> None:
    client, task = protected_server
    client.set_cookie("rdagent_auth", "secret-token")
    response = client.post(
        "/upload",
        headers={"Authorization": authorization, "Origin": "http://localhost"},
    )
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    task.assert_not_called()


@pytest.mark.offline
def test_origin_cannot_be_overridden_by_referer_or_forwarded_headers(
    protected_server: tuple[FlaskClient, Mock],
) -> None:
    client, task = protected_server
    response = client.post(
        "/upload",
        headers={
            "Authorization": "Bearer secret-token",
            "Origin": "https://evil.example",
            "Referer": "http://localhost/",
            "X-Forwarded-Host": "evil.example",
            "X-Forwarded-Proto": "https",
        },
    )
    assert response.status_code == HTTPStatus.FORBIDDEN
    task.assert_not_called()


@pytest.mark.offline
def test_internal_receive_accepts_bearer_without_browser_headers(protected_server: tuple[FlaskClient, Mock]) -> None:
    client, task = protected_server
    response = client.post(
        "/receive",
        json={"id": "trace", "msg": {"tag": "test"}},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == HTTPStatus.OK
    task.return_value.messages.append.assert_called_once_with({"tag": "test"})


@pytest.mark.offline
@pytest.mark.parametrize("count", [0, 2])
def test_general_model_requires_one_file_before_side_effects(
    protected_server: tuple[FlaskClient, Mock],
    tmp_path: Path,
    count: int,
) -> None:
    client, task = protected_server
    response = client.post(
        "/upload",
        data={
            "scenario": "General Model Implementation",
            "files": [(BytesIO(b"%PDF-test"), f"report{i}.pdf") for i in range(count)],
        },
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    task.assert_not_called()
    assert not list(tmp_path.iterdir())
