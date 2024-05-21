from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Sequence

from loguru import logger


if TYPE_CHECKING:
    from loguru import Logger


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
    def get_all_colors(cls: type[LogColors]) -> list:
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
            # Changes to accommodate ruff checks.
            # Original code:
            # raise ValueError(f"color should be in: {colors} but now is: {color}")
            # Description of the problem:
            # TRY003 Avoid specifying long messages outside the exception class
            # EM102 Exception must not use an f-string literal, assign to variable first
            # References:
            # https://docs.astral.sh/ruff/rules/raise-vanilla-args/
            # https://docs.astral.sh/ruff/rules/f-string-in-exception/
            error_message = f"color should be in: {colors} but now is: {color}"
            raise ValueError(error_message)
        if style and style in colors:
            # Changes to accommodate ruff checks.
            # Original code:
            # raise ValueError(f"style should be in: {colors} but now is: {style}")
            # Description of the problem:
            # TRY003 Avoid specifying long messages outside the exception class
            # EM102 Exception must not use an f-string literal, assign to variable first
            # References:
            # https://docs.astral.sh/ruff/rules/raise-vanilla-args/
            # https://docs.astral.sh/ruff/rules/f-string-in-exception/
            error_message = f"style should be in: {colors} but now is: {style}"
            raise ValueError(error_message)

        text = f"{color}{text}{self.END}"

        return f"{style}{text}{self.END}"


class FinCoLog:
    # logger.add(loguru_handler, level="INFO")   # you can add use storage as a loguru handler

    def __init__(self) -> None:
        self.logger: Logger = logger

    def info(self, *args: Sequence, plain: bool = False, title: str = "Info") -> None:
        if plain:
            return self.plain_info(*args)
        for arg in args:
            # Changes to accommodate ruff checks.
            # Original code:
            # self.logger.info(f"{LogColors.WHITE}{arg}{LogColors.END}")
            # Description of the problem:
            # G004 Logging statement uses f-string
            # References:
            # https://docs.astral.sh/ruff/rules/logging-f-string/
            info = f"{LogColors.WHITE}{arg}{LogColors.END}"
            self.logger.info(info)
        return None

    def __getstate__(self) -> dict:
        return {}

    # Changes to accommodate ruff checks.
    # Original code: def __setstate__(self, _: str) -> None:
    # Description of the problem:
    # PLE0302 The special method `__setstate__` expects 2 parameters, 1 was given
    # References: https://docs.astral.sh/ruff/rules/unexpected-special-method-signature/
    def __setstate__(self, _: str) -> None:
        self.logger = logger

    def plain_info(self, *args: Sequence) -> None:
        for arg in args:
            # Changes to accommodate ruff checks.
            # Original code:
            # self.logger.info(
            #   f"""
            #   {LogColors.YELLOW}{LogColors.BOLD}
            #   Info:{LogColors.END}{LogColors.WHITE}{arg}{LogColors.END}
            #   """,
            # )
            # Description of the problem:
            # G004 Logging statement uses f-string
            # References:
            # https://docs.astral.sh/ruff/rules/logging-f-string/
            info = f"""
                {LogColors.YELLOW}{LogColors.BOLD}
                Info:{LogColors.END}{LogColors.WHITE}{arg}{LogColors.END}
            """
            self.logger.info(info)

    def warning(self, *args: Sequence) -> None:
        for arg in args:
            # Changes to accommodate ruff checks.
            # Original code:
            # self.logger.warning(
            #   f"{LogColors.BLUE}{LogColors.BOLD}Warning:{LogColors.END}{arg}",
            # )
            # Description of the problem:
            # G004 Logging statement uses f-string
            # References:
            # https://docs.astral.sh/ruff/rules/logging-f-string/
            info = f"{LogColors.BLUE}{LogColors.BOLD}Warning:{LogColors.END}{arg}"
            self.logger.warning(info)

    def error(self, *args: Sequence) -> None:
        for arg in args:
            # Changes to accommodate ruff checks.
            # Original code:
            # self.logger.error(
            #   f"{LogColors.RED}{LogColors.BOLD}Error:{LogColors.END}{arg}",
            # )
            # Description of the problem:
            # G004 Logging statement uses f-string
            # References:
            # https://docs.astral.sh/ruff/rules/logging-f-string/
            info = f"{LogColors.RED}{LogColors.BOLD}Error:{LogColors.END}{arg}"
            self.logger.error(info)
