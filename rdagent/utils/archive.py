import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_MAX_ARCHIVE_MEMBERS = 10_000
DEFAULT_MAX_UNCOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024


def _safe_destination(root: Path, member_name: str) -> Path:
    member_path = PurePosixPath(member_name.replace("\\", "/"))
    if member_path.is_absolute() or PureWindowsPath(member_name).drive or ".." in member_path.parts:
        message = f"Unsafe archive member path: {member_name}"
        raise ValueError(message)
    destination = root.joinpath(*member_path.parts).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        message = f"Archive member escapes destination: {member_name}"
        raise ValueError(message) from exc
    return destination


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> None:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > max_members or sum(member.file_size for member in members) > max_uncompressed_bytes:
            message = "Archive exceeds configured extraction limits"
            raise ValueError(message)

        targets: list[tuple[zipfile.ZipInfo, Path]] = []
        for member in members:
            target = _safe_destination(root, member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                message = f"Archive links are not allowed: {member.filename}"
                raise ValueError(message)
            file_type = stat.S_IFMT(mode)
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                message = f"Unsupported archive member type: {member.filename}"
                raise ValueError(message)
            targets.append((member, target))

        for member, target in targets:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def safe_extract_tar(
    archive_path: str | Path,
    destination: str | Path,
    *,
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> None:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        if len(members) > max_members or sum(member.size for member in members) > max_uncompressed_bytes:
            message = "Archive exceeds configured extraction limits"
            raise ValueError(message)

        targets: list[tuple[tarfile.TarInfo, Path]] = []
        for member in members:
            target = _safe_destination(root, member.name)
            if not member.isdir() and not member.isfile():
                message = f"Unsupported archive member type: {member.name}"
                raise ValueError(message)
            targets.append((member, target))

        for member, target in targets:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = archive.extractfile(member)
            if source is None:
                message = f"Unable to read archive member: {member.name}"
                raise ValueError(message)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
