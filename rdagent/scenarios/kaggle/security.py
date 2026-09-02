import re

_COMPETITION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")


def validate_competition_slug(competition: str) -> str:
    if not _COMPETITION_RE.fullmatch(competition):
        message = "Competition must be a lowercase Kaggle slug containing only letters, digits, and hyphens"
        raise ValueError(message)
    return competition
