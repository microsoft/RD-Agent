from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import pytest

import rdagent.log.server.app as server
from rdagent.log.server.security import (
    parse_competition,
    resolve_within,
    validate_scenario,
    validate_upload_filename,
)


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
def test_upload_rejects_unknown_scenario_before_writing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "")
    monkeypatch.setattr(server, "upload_folder_path", tmp_path / "uploads")
    client = server.app.test_client()

    response = client.post("/upload", data={"scenario": "../Data Science"})

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


@pytest.mark.offline
def test_upload_rejects_pickle_before_starting_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(server.app.config, "AUTH_TOKEN", "")
    monkeypatch.setattr(server, "upload_folder_path", tmp_path / "uploads")
    monkeypatch.setattr(server, "log_folder_path", tmp_path / "traces")
    client = server.app.test_client()

    response = client.post(
        "/upload",
        data={"scenario": "Finance Data Building", "files": (BytesIO(b"payload"), "payload.pkl")},
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert not (tmp_path / "uploads").exists()
