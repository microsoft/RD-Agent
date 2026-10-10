import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

GRADE_PATH = (
    Path(__file__).resolve().parents[3]
    / "rdagent"
    / "scenarios"
    / "data_science"
    / "example"
    / "eval"
    / "arf-12-hours-prediction-task"
    / "grade.py"
)


def _load_grade_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("arf_12_hours_grade", GRADE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.offline
def test_arf_grade_rejects_submission_missing_target_column() -> None:
    grade_module = _load_grade_module()
    submission = pd.DataFrame({"ID": [1, 2]})
    answers = pd.DataFrame({"ID": [1, 2], "ARF_LABEL": [0, 1]})

    with pytest.raises(Exception, match="ARF_LABEL") as exc_info:
        grade_module.grade(submission, answers)
    assert type(exc_info.value).__name__ == "InvalidSubmissionError"
