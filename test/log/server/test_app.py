from pathlib import Path

from rdagent.log.server.app import _load_existing_traces


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
