from pathlib import Path

from rdagent.log.server.app import app, _collect_existing_trace_ids, _load_existing_traces


def test_load_existing_traces_only_loads_trace_dirs_with_pkl(tmp_path, monkeypatch):
    trace_root = tmp_path / "traces"
    active_trace = trace_root / "scenario_a" / "trace_1"
    empty_trace = trace_root / "scenario_a" / "trace_2"
    nested_upload = trace_root / "uploads" / "scenario_b" / "trace_3"

    (active_trace / "feedback").mkdir(parents=True)
    empty_trace.mkdir(parents=True)
    (nested_upload / "feedback").mkdir(parents=True)

    (active_trace / "feedback" / "0001.pkl").write_bytes(b"trace")
    (nested_upload / "feedback" / "0001.pkl").write_bytes(b"upload-trace")

    loaded: list[tuple[Path, str]] = []

    def fake_read_trace(log_path: Path, id: str = "") -> None:
        loaded.append((log_path, id))

    monkeypatch.setattr("rdagent.log.server.app.read_trace", fake_read_trace)

    _load_existing_traces(trace_root)

    assert loaded == [(active_trace, str(active_trace))]


def test_collect_existing_trace_ids_ignores_uploads(tmp_path):
    trace_root = tmp_path / "traces"
    active_trace = trace_root / "scenario_a" / "trace_1"
    uploads_trace = trace_root / "uploads" / "scenario_b" / "trace_2"

    (active_trace / "feedback").mkdir(parents=True)
    (uploads_trace / "feedback").mkdir(parents=True)

    (active_trace / "feedback" / "0001.pkl").write_bytes(b"trace")
    (uploads_trace / "feedback" / "0001.pkl").write_bytes(b"upload-trace")

    assert _collect_existing_trace_ids(trace_root) == ["scenario_a/trace_1"]


def test_traces_route_returns_visible_trace_ids(tmp_path, monkeypatch):
    trace_root = tmp_path / "traces"
    active_trace = trace_root / "scenario_a" / "trace_1"
    uploads_trace = trace_root / "uploads" / "scenario_b" / "trace_2"

    (active_trace / "feedback").mkdir(parents=True)
    (uploads_trace / "feedback").mkdir(parents=True)

    (active_trace / "feedback" / "0001.pkl").write_bytes(b"trace")
    (uploads_trace / "feedback" / "0001.pkl").write_bytes(b"upload-trace")

    monkeypatch.setattr("rdagent.log.server.app.log_folder_path", trace_root)

    with app.test_client() as client:
        response = client.get("/traces")

    assert response.status_code == 200
    assert response.get_json() == ["scenario_a/trace_1"]
