from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock

import pytest
from filelock import FileLock

from rdagent.core import utils
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.serialization import loads

pytestmark = pytest.mark.offline
WAIT_TIMEOUT = 10
POST_PROCESS_INCREMENT = 10
EXPECTED_BYPASS_CALLS = 2


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(RD_AGENT_SETTINGS, "pickle_cache_folder_path_str", str(tmp_path))
    monkeypatch.setattr(RD_AGENT_SETTINGS, "cache_with_pickle", True)
    monkeypatch.setattr(RD_AGENT_SETTINGS, "use_file_lock", True)
    monkeypatch.setattr(RD_AGENT_SETTINGS, "artifact_signing_key", "test-pickle-cache-signing-key-not-a-secret")
    monkeypatch.setattr(RD_AGENT_SETTINGS, "allow_unsafe_legacy_pickle", False)
    return tmp_path


@pytest.mark.parametrize("post_process", [False, True])
def test_concurrent_cache_misses_compute_once(
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    post_process: bool,
) -> None:
    first_started = Event()
    second_lock_attempted = Event()
    counter_lock = Lock()
    lock_attempts = 0
    calls = 0

    @contextmanager
    def observed_lock(path: Path) -> Iterator[FileLock]:
        nonlocal lock_attempts
        with counter_lock:
            lock_attempts += 1
            if lock_attempts > 1:
                second_lock_attempted.set()
        # Keep the actual filesystem lock; only observe the second attempt.
        with FileLock(path) as lock:
            yield lock

    monkeypatch.setattr(utils, "FileLock", observed_lock)

    def process_cached(*, cached_res: int) -> int:
        return cached_res + POST_PROCESS_INCREMENT

    @utils.cache_with_pickle(lambda: "same-key", process_cached if post_process else None)
    def compute() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            assert second_lock_attempted.wait(WAIT_TIMEOUT)
        return calls

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(compute)
        assert first_started.wait(WAIT_TIMEOUT)
        second = executor.submit(compute)
        assert first.result(timeout=WAIT_TIMEOUT) == 1
        assert second.result(timeout=WAIT_TIMEOUT) == (1 + POST_PROCESS_INCREMENT if post_process else 1)

    assert calls == 1
    cache_file = cache_dir / f"{compute.__module__}.{compute.__name__}" / "same-key.pkl"
    assert loads(cache_file.read_bytes()) == 1


@pytest.mark.parametrize("use_file_lock", [False, True])
def test_cached_none_is_a_hit(cache_dir: Path, monkeypatch: pytest.MonkeyPatch, *, use_file_lock: bool) -> None:
    assert cache_dir.is_dir()
    monkeypatch.setattr(RD_AGENT_SETTINGS, "use_file_lock", use_file_lock)
    calls = 0

    @utils.cache_with_pickle(lambda: "none-result")
    def compute() -> None:
        nonlocal calls
        calls += 1

    assert compute() is None
    assert compute() is None
    assert calls == 1


def test_failed_computation_releases_lock(cache_dir: Path) -> None:
    calls = 0

    @utils.cache_with_pickle(lambda: "retry")
    def compute() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            message = "computation failed"
            raise ValueError(message)
        return "recovered"

    with pytest.raises(ValueError, match="computation failed"):
        compute()
    lock_file = cache_dir / f"{compute.__module__}.{compute.__name__}" / "retry.lock"
    with FileLock(lock_file, timeout=0):
        pass
    assert compute() == "recovered"
    assert compute() == "recovered"


def test_cached_result_postprocessing_runs_after_unlock(cache_dir: Path) -> None:
    def process_cached(*, cached_res: str) -> str:
        lock_file = cache_dir / f"{compute.__module__}.{compute.__name__}" / "postprocess.lock"
        with FileLock(lock_file, timeout=0):
            return f"{cached_res}:cached"

    @utils.cache_with_pickle(lambda: "postprocess", process_cached)
    def compute() -> str:
        return "result"

    assert compute() == "result"
    assert compute() == "result:cached"


@pytest.mark.parametrize("bypass", ["disabled", "no-key"])
def test_cache_bypass_does_not_acquire_lock(cache_dir: Path, monkeypatch: pytest.MonkeyPatch, bypass: str) -> None:
    assert cache_dir.is_dir()
    monkeypatch.setattr(RD_AGENT_SETTINGS, "cache_with_pickle", bypass != "disabled")

    def unexpected_lock(_path: Path) -> None:
        pytest.fail("A bypassed cache must not acquire a lock")

    monkeypatch.setattr(utils, "FileLock", unexpected_lock)
    calls = []

    @utils.cache_with_pickle(lambda: None if bypass == "no-key" else "key")
    def compute() -> list:
        calls.append(None)
        return calls

    assert compute() is calls
    assert compute() is calls
    assert len(calls) == EXPECTED_BYPASS_CALLS
