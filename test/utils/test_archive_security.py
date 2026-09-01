import io
import tarfile
import zipfile
from pathlib import Path

import pytest
from rdagent.scenarios.kaggle.security import validate_competition_slug
from rdagent.utils.archive import safe_extract_tar, safe_extract_zip


@pytest.mark.offline
def test_competition_slug_rejects_command_and_path_syntax() -> None:
    assert validate_competition_slug("aerial-cactus-identification") == "aerial-cactus-identification"
    for competition in ("../../tmp", "safe;touch-pwned", "UPPERCASE", "with space"):
        with pytest.raises(ValueError, match="Competition"):
            validate_competition_slug(competition)


@pytest.mark.offline
def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "payload.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escaped.txt", "payload")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract_zip(archive_path, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.offline
def test_safe_extract_zip_rejects_windows_absolute_path(tmp_path: Path) -> None:
    archive_path = tmp_path / "payload.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("C:\\escaped.txt", "payload")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract_zip(archive_path, tmp_path / "output")


@pytest.mark.offline
def test_safe_extract_zip_extracts_regular_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/result.txt", "safe content")

    output = tmp_path / "output"
    safe_extract_zip(archive_path, output)
    assert (output / "nested" / "result.txt").read_text() == "safe content"


@pytest.mark.offline
def test_safe_extract_tar_rejects_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "payload.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../escaped.txt"
        archive.addfile(member)

    with pytest.raises(ValueError, match="Unsupported archive member type"):
        safe_extract_tar(archive_path, tmp_path / "output")


@pytest.mark.offline
def test_safe_extract_tar_extracts_regular_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "safe.tar"
    content = b"safe content"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("nested/result.txt")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))

    output = tmp_path / "output"
    safe_extract_tar(archive_path, output)
    assert (output / "nested" / "result.txt").read_bytes() == content
