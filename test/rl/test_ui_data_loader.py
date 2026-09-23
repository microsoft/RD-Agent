import pickle
from pathlib import Path

import pytest

from rdagent.app.rl.ui.data_loader import load_session


@pytest.mark.offline
def test_load_session_skips_unsigned_legacy_pickles(tmp_path: Path) -> None:
    # Pickles written before artifact signing are rejected with UntrustedArtifactError
    # (a ValueError), which load_session is meant to skip.
    run_dir = tmp_path / "Loop_0" / "running"
    run_dir.mkdir(parents=True)
    (run_dir / "2026-09-01_10-00-00-000000.pkl").write_bytes(pickle.dumps({"x": 1}))

    session = load_session(tmp_path)

    assert session.loops == {}
