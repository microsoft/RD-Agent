import re

EDA_PART_PATTERN = re.compile(
    r"=== Start of EDA part ===(.*?)=== End of EDA part ===",
    flags=re.DOTALL,
)


def extract_eda_parts(stdout: str) -> list[str]:
    """Extract every explicitly delimited EDA block from stdout."""
    return EDA_PART_PATTERN.findall(stdout)


def remove_eda_part(stdout: str) -> str:
    """Data Science scenario have a LLM-based EDA feature. We can remove it when current task does not involve EDA"""
    return EDA_PART_PATTERN.sub("", stdout)
