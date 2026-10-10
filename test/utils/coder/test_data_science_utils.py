import pytest

from rdagent.components.coder.data_science.utils import (
    extract_eda_parts,
    remove_eda_part,
)


@pytest.mark.offline
def test_remove_eda_part_preserves_text_between_multiple_blocks() -> None:
    stdout = (
        "before\n"
        "=== Start of EDA part ===\n"
        "eda one\n"
        "=== End of EDA part ===\n"
        "between\n"
        "=== Start of EDA part ===\n"
        "eda two\n"
        "=== End of EDA part ===\n"
        "after\n"
    )

    cleaned = remove_eda_part(stdout)

    assert cleaned == "before\n\nbetween\n\nafter\n"


@pytest.mark.offline
def test_extract_eda_parts_returns_all_blocks() -> None:
    stdout = (
        "before\n"
        "=== Start of EDA part ===\n"
        "eda one\n"
        "=== End of EDA part ===\n"
        "between\n"
        "=== Start of EDA part ===\n"
        "eda two\n"
        "=== End of EDA part ===\n"
        "after\n"
    )

    assert extract_eda_parts(stdout) == ["\neda one\n", "\neda two\n"]


@pytest.mark.offline
@pytest.mark.parametrize(
    ("stdout", "eda_parts", "cleaned_stdout"),
    [
        ("", [], ""),
        ("normal output\n", [], "normal output\n"),
        (
            "before=== Start of EDA part ===first\nsecond=== End of EDA part ===after",
            ["first\nsecond"],
            "beforeafter",
        ),
        ("before=== Start of EDA part ====== End of EDA part ===after", [""], "beforeafter"),
        (
            "=== Start of EDA part ===one=== End of EDA part ===" "=== Start of EDA part ===two=== End of EDA part ===",
            ["one", "two"],
            "",
        ),
        ("before=== Start of EDA part ===unfinished", [], "before=== Start of EDA part ===unfinished"),
        ("before=== End of EDA part ===after", [], "before=== End of EDA part ===after"),
        (
            "=== Start of EDA part ===complete=== End of EDA part ===" "between=== Start of EDA part ===unfinished",
            ["complete"],
            "between=== Start of EDA part ===unfinished",
        ),
    ],
    ids=["empty-stdout", "no-eda", "multiline", "empty-eda", "adjacent", "unclosed", "unopened", "trailing-unclosed"],
)
def test_eda_parts(stdout: str, eda_parts: list[str], cleaned_stdout: str) -> None:
    assert extract_eda_parts(stdout) == eda_parts
    assert remove_eda_part(stdout) == cleaned_stdout
