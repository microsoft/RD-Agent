"""
Tools that support generating better formats.
"""


def shrink_text(text: str, context_lines: int = 200) -> str:
    """
    When the context is too long, hide the part that is not important.

        text before
        ... (XXXXX lines are hidden) ...
        text after
    """
    lines = text.splitlines()
    total_lines = len(lines)

    if total_lines <= context_lines:
        return text

    # Calculate how many lines to show from start and end
    half_lines = context_lines // 2
    start = "\n".join(lines[:half_lines])
    end = "\n".join(lines[-half_lines:])

    # Count the number of lines we're hiding
    hidden_lines = total_lines - half_lines * 2

    return f"{start}\n... ({hidden_lines} lines are hidden) ...\n{end}"
