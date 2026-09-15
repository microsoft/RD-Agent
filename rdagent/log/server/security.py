import re
from pathlib import Path

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
