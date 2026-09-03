import contextlib
import hashlib
import hmac
import os
import pickle
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from filelock import FileLock

from rdagent.core.conf import RD_AGENT_SETTINGS

_MAGIC = b"RDAGENT_SIGNED_PICKLE_V1\n"
_DIGEST_SIZE = hashlib.sha256().digest_size
_MIN_KEY_BYTES = 32


class PickleSerializer(Protocol):
    def dumps(self, obj: object, protocol: int | None = None) -> bytes: ...

    def loads(self, data: bytes) -> Any: ...


class UntrustedArtifactError(ValueError):
    """Raised before deserialization when an artifact is unsigned or has been modified."""


@lru_cache(maxsize=8)
def _signing_key_from_config(configured_key: str, configured_path: str) -> bytes:
    if configured_key:
        if len(configured_key.encode()) < _MIN_KEY_BYTES:
            message = "ARTIFACT_SIGNING_KEY must contain at least 32 bytes"
            raise ValueError(message)
        return hashlib.sha256(configured_key.encode()).digest()

    key_path = Path(configured_path).expanduser().resolve()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{key_path}.lock"):
        try:
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            key = key_path.read_bytes()
        else:
            key = secrets.token_bytes(32)
            with os.fdopen(descriptor, "wb") as key_file:
                key_file.write(key)
    if len(key) < _MIN_KEY_BYTES:
        message = f"Artifact signing key at {key_path} must contain at least 32 bytes"
        raise ValueError(message)
    with contextlib.suppress(OSError):
        key_path.chmod(0o600)
    return key


def _signing_key() -> bytes:
    return _signing_key_from_config(
        RD_AGENT_SETTINGS.artifact_signing_key,
        str(RD_AGENT_SETTINGS.artifact_signing_key_path),
    )


def dumps(obj: object, *, serializer: PickleSerializer = pickle) -> bytes:
    payload = serializer.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hmac.digest(_signing_key(), payload, "sha256")
    return _MAGIC + digest + payload


def loads(
    data: bytes,
    *,
    serializer: PickleSerializer = pickle,
    allow_unsafe_legacy: bool | None = None,
) -> Any:
    if data.startswith(_MAGIC):
        digest_start = len(_MAGIC)
        digest_end = digest_start + _DIGEST_SIZE
        digest = data[digest_start:digest_end]
        payload = data[digest_end:]
        expected = hmac.digest(_signing_key(), payload, "sha256")
        if len(digest) != _DIGEST_SIZE or not hmac.compare_digest(digest, expected):
            message = "Artifact signature verification failed"
            raise UntrustedArtifactError(message)
        return serializer.loads(payload)

    allow_legacy = RD_AGENT_SETTINGS.allow_unsafe_legacy_pickle if allow_unsafe_legacy is None else allow_unsafe_legacy
    if not allow_legacy:
        message = "Unsigned legacy pickle rejected; enable ALLOW_UNSAFE_LEGACY_PICKLE only for trusted artifacts"
        raise UntrustedArtifactError(message)
    return serializer.loads(data)


def dump(obj: object, file: BinaryIO, *, serializer: PickleSerializer = pickle) -> None:
    file.write(dumps(obj, serializer=serializer))


def load(
    file: BinaryIO,
    *,
    serializer: PickleSerializer = pickle,
    allow_unsafe_legacy: bool | None = None,
) -> Any:
    return loads(file.read(), serializer=serializer, allow_unsafe_legacy=allow_unsafe_legacy)
