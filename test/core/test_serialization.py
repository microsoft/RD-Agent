import pickle
from pathlib import Path

import dill
import pytest

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.serialization import UntrustedArtifactError, dumps, loads
from rdagent.core.utils import cache_with_pickle
from rdagent.log.storage import FileStorage

KEY_FILE_MODE = 0o600
EXPECTED_INCREMENTED_VALUE = 2


@pytest.fixture
def signing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    key_path = tmp_path / "artifact.key"
    monkeypatch.setattr(RD_AGENT_SETTINGS, "artifact_signing_key", "")
    monkeypatch.setattr(RD_AGENT_SETTINGS, "artifact_signing_key_path", key_path)
    monkeypatch.setattr(RD_AGENT_SETTINGS, "allow_unsafe_legacy_pickle", False)
    return key_path


@pytest.mark.offline
def test_signed_pickle_round_trip(signing_key: Path) -> None:
    value = {"score": 0.75, "items": [1, 2, 3]}
    assert loads(dumps(value)) == value
    assert signing_key.stat().st_mode & 0o777 == KEY_FILE_MODE
    assert "artifact_signing_key" not in RD_AGENT_SETTINGS.model_dump()


@pytest.mark.offline
def test_configured_signing_key_does_not_create_key_file(signing_key: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RD_AGENT_SETTINGS, "artifact_signing_key", "shared-secret-with-sufficient-entropy")
    assert loads(dumps({"shared": True})) == {"shared": True}
    assert not signing_key.exists()


@pytest.mark.offline
def test_short_configured_signing_key_is_rejected(signing_key: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert signing_key.parent.exists()
    monkeypatch.setattr(RD_AGENT_SETTINGS, "artifact_signing_key", "too-short")
    with pytest.raises(ValueError, match="at least 32 bytes"):
        dumps({"shared": True})


@pytest.mark.offline
def test_signed_dill_round_trip(signing_key: Path) -> None:
    assert signing_key.parent.exists()

    def increment(item: int) -> int:
        return item + 1

    restored = loads(dumps(increment, serializer=dill), serializer=dill)
    assert restored(1) == EXPECTED_INCREMENTED_VALUE


@pytest.mark.offline
def test_modified_signed_pickle_is_rejected(signing_key: Path) -> None:
    assert signing_key.parent.exists()
    payload = bytearray(dumps({"safe": True}))
    payload[-1] ^= 1
    with pytest.raises(UntrustedArtifactError, match="signature"):
        loads(bytes(payload))


@pytest.mark.offline
def test_unsigned_pickle_is_rejected_before_deserialization(signing_key: Path, tmp_path: Path) -> None:
    assert signing_key.parent.exists()
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self) -> tuple:
            return marker.touch, ()

    payload = pickle.dumps(Payload())
    with pytest.raises(UntrustedArtifactError, match="Unsigned"):
        loads(payload)
    assert not marker.exists()


@pytest.mark.offline
def test_legacy_pickle_requires_explicit_opt_in(signing_key: Path) -> None:
    assert signing_key.parent.exists()
    payload = pickle.dumps({"legacy": True})
    assert loads(payload, allow_unsafe_legacy=True) == {"legacy": True}


@pytest.mark.offline
def test_unsigned_cache_is_treated_as_cache_miss(
    signing_key: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert signing_key.parent.exists()
    monkeypatch.setattr(RD_AGENT_SETTINGS, "pickle_cache_folder_path_str", str(tmp_path / "cache"))
    calls = 0

    @cache_with_pickle(lambda: "fixed-key", force=True)
    def compute() -> str:
        nonlocal calls
        calls += 1
        return "fresh"

    cache_file = tmp_path / "cache" / f"{compute.__module__}.{compute.__name__}" / "fixed-key.pkl"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(pickle.dumps("attacker-controlled"))

    assert compute() == "fresh"
    assert calls == 1
    assert loads(cache_file.read_bytes()) == "fresh"


@pytest.mark.offline
def test_file_storage_writes_and_reads_signed_messages(signing_key: Path, tmp_path: Path) -> None:
    assert signing_key.parent.exists()
    storage = FileStorage(tmp_path / "trace")
    artifact_path = storage.log({"message": "safe"}, tag="record.test")

    assert isinstance(artifact_path, Path)
    assert artifact_path.read_bytes().startswith(b"RDAGENT_SIGNED_PICKLE_V1")
    messages = list(storage.iter_msg(tag="record.test"))
    assert [message.content for message in messages] == [{"message": "safe"}]
