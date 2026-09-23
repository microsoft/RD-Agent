import re
from pathlib import Path
from urllib.parse import urlsplit

SCENARIO_TARGETS = {
    "Finance Data Building": "fin_factor",
    "Finance Model Implementation": "fin_model",
    "Finance Whole Pipeline": "fin_quant",
    "Finance Data Building (Reports)": "fin_factor_report",
    "General Model Implementation": "general_model",
    "Data Science": "data_science",
}

_COMPETITION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")
_UNSAFE_UPLOAD_SUFFIXES = {".dill", ".pickle", ".pkl", ".py", ".pyc", ".pyo"}
_ERR_COMPETITION_PREFIX = "Competition must start with 'MLE-Bench:'"
_ERR_INVALID_COMPETITION = "Invalid competition name"
_ERR_INVALID_FILENAME = "Invalid upload filename"
_ERR_PATH_ESCAPE = "Path escapes the configured root"
_ERR_UNKNOWN_SCENARIO = "Unknown scenario"
_ERR_UNSAFE_FILE_TYPE = "Unsafe upload file type"


def normalize_origin(value: str, *, allow_path: bool = False) -> str | None:
    """Parse a single HTTP origin, optionally extracting it from a Referer URL."""
    if not value or any(char.isspace() for char in value) or "\\" in value or "*" in value:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        if not allow_path and (parsed.path or parsed.query or parsed.fragment):
            return None
        port = parsed.port
    except ValueError:
        return None
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port != {"http": 80, "https": 443}[parsed.scheme]:
        host = f"{host}:{port}"
    return f"{parsed.scheme}://{host}"


def validate_scenario(value: str | None) -> str:
    if value not in SCENARIO_TARGETS:
        raise ValueError(_ERR_UNKNOWN_SCENARIO)
    return value


def parse_competition(value: str | None) -> str:
    prefix = "MLE-Bench:"
    if value is None or not value.startswith(prefix):
        raise ValueError(_ERR_COMPETITION_PREFIX)
    competition = value[len(prefix) :]
    if not _COMPETITION_RE.fullmatch(competition):
        raise ValueError(_ERR_INVALID_COMPETITION)
    return competition


def resolve_within(root: str | Path, *parts: str) -> Path:
    resolved_root = Path(root).resolve()
    resolved_path = resolved_root.joinpath(*parts).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(_ERR_PATH_ESCAPE) from exc
    return resolved_path


def validate_upload_filename(value: str) -> str:
    filename = Path(value).name
    if not filename or filename in {".", ".."}:
        raise ValueError(_ERR_INVALID_FILENAME)
    if Path(filename).suffix.lower() in _UNSAFE_UPLOAD_SUFFIXES:
        raise ValueError(_ERR_UNSAFE_FILE_TYPE)
    return filename
