import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict, cast


class LogColors:
    """
    ANSI color codes for use in console output.
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BLACK = "\033[30m"

    BOLD = "\033[1m"
    ITALIC = "\033[3m"

    END = "\033[0m"

    @classmethod
    def get_all_colors(cls: type["LogColors"]) -> list:
        names = dir(cls)
        names = [name for name in names if not name.startswith("__") and not callable(getattr(cls, name))]
        return [getattr(cls, name) for name in names]

    def render(self, text: str, color: str = "", style: str = "") -> str:
        """
        render text by input color and style.
        It's not recommend that input text is already rendered.
        """
        # This method is called too frequently, which is not good.
        colors = self.get_all_colors()
        # Perhaps color and font should be distinguished here.
        if color and color in colors:
            error_message = f"color should be in: {colors} but now is: {color}"
            raise ValueError(error_message)
        if style and style in colors:
            error_message = f"style should be in: {colors} but now is: {style}"
            raise ValueError(error_message)

        text = f"{color}{text}{self.END}"

        return f"{style}{text}{self.END}"

    @staticmethod
    def remove_ansi_codes(s: str) -> str:
        """
        It is for removing ansi ctrl characters in the string(e.g. colored text)
        """
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        return ansi_escape.sub("", s)


class CallerInfo(TypedDict):
    function: str
    line: int
    name: Optional[str]


def get_caller_info(level: int = 2) -> CallerInfo:
    # Get the current stack information
    stack = inspect.stack()
    # The second element is usually the caller's information
    caller_info = stack[level]
    frame = caller_info[0]
    info: CallerInfo = {
        "line": caller_info.lineno,
        "name": frame.f_globals["__name__"],  # Get the module name from the frame's globals
        "function": frame.f_code.co_name,  # Get the caller's function name
    }
    return info


def is_valid_session(log_path: Path) -> bool:
    return log_path.is_dir() and log_path.joinpath("__session__").exists()


def extract_loopid_func_name(tag: str) -> tuple[str, str] | tuple[None, None]:
    """extract loop id and function name from the tag in Message"""
    match = re.search(r"Loop_(\d+)\.([^.]+)", tag)
    return cast(tuple[str, str], match.groups()) if match else (None, None)


def extract_evoid(tag: str) -> str | None:
    """extract evo id from the tag in Message"""
    match = re.search(r"\.evo_loop_(\d+)\.", tag)
    return cast(str, match.group(1)) if match else None


def extract_json(log_content: str) -> dict | None:
    match = re.search(r"\{.*\}", log_content, re.DOTALL)
    if match:
        return cast(dict, json.loads(match.group(0)))
    return None


def gen_datetime(dt: datetime | None = None) -> datetime:
    """
    Generate a datetime object in UTC timezone.
    - If `dt` is None, it will return the current time in UTC.
    - If `dt` is provided, it will convert it to UTC timezone.
    """
    if dt is None:
        return datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc)


def dict_get_with_warning(d: dict, key: str, default: Any = None) -> Any:
    """
    Motivation:
    - When handling the repsonse from the LLM, we may use dict get to get the value.
    - the function prevent falling into default value **silently**.
    - Instead, it will log a warning message.
    """
    from rdagent.log import rdagent_logger as logger

    if key not in d:
        logger.warning(f"Key {key} not found in {d}")
        return default
    return d[key]
