import pytest

from rdagent.components.coder.data_science.utils import remove_eda_part


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
