import re


def remove_eda_part(stdout: str) -> str:
    """Data Science scenario have a LLM-based EDA feature. We can remove it when current task does not involve EDA"""
    return re.sub(r"=== Start of EDA part ===(.*)=== End of EDA part ===", "", stdout, flags=re.DOTALL)
