from pathlib import Path

import pytest

from rdagent.components.coder.finetune.unified_validator import LLMConfigValidator
from rdagent.scenarios.data_science.proposal.exp_gen.select.submit import _parsing_score

EXPECTED_SCORE = 0.75


@pytest.mark.offline
def test_score_parser_accepts_finite_json_number() -> None:
    assert _parsing_score('result: {"score": 0.75}') == EXPECTED_SCORE


@pytest.mark.offline
def test_score_parser_does_not_execute_python(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    payload = f'{{"score": __import__("pathlib").Path("{marker}").touch()}}'

    assert _parsing_score(payload) is None
    assert not marker.exists()


@pytest.mark.offline
@pytest.mark.parametrize("score", ["NaN", "Infinity", "true", '"1.0"'])
def test_score_parser_rejects_non_finite_or_non_numeric_values(score: str) -> None:
    assert _parsing_score(f'{{"score": {score}}}') is None


@pytest.mark.offline
def test_training_metrics_parser_does_not_execute_python(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    stdout = f"Running training\n{{'train_runtime': __import__('pathlib').Path('{marker}').touch()}}"
    validator = object.__new__(LLMConfigValidator)

    parsed = validator._parse_execution_log(stdout, 0)  # noqa: SLF001

    assert str(marker) not in parsed
    assert not marker.exists()
