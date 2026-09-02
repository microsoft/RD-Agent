from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rdagent.components.coder.finetune.unified_validator import LLMConfigValidator
from rdagent.scenarios.data_science.proposal.exp_gen.select.submit import (
    ValidationSelector,
    _parsing_score,
)

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
@pytest.mark.parametrize(
    ("stdout", "is_valid"),
    [
        ('{"score": 0.75, "metric": "auc"}', True),
        ("{'score': 0.75, 'metric': 'auc'}", False),
    ],
)
def test_reusable_grade_script_requires_strict_json(
    stdout: str, is_valid: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,  # noqa: FBT001
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    (tmp_path / "submission.csv").touch()

    workspace = SimpleNamespace(
        workspace_path=workspace_path,
        inject_code_from_file_dict=Mock(),
        inject_files=Mock(),
        run=Mock(return_value=SimpleNamespace(exit_code=0, stdout=stdout)),
    )
    monkeypatch.setattr("rdagent.scenarios.data_science.proposal.exp_gen.select.submit.FBWorkspace", lambda: workspace)
    monkeypatch.setattr(
        "rdagent.scenarios.data_science.proposal.exp_gen.select.submit.get_ds_env", lambda **_: object(),
    )

    selector = object.__new__(ValidationSelector)
    reference_exp = SimpleNamespace(experiment_workspace=SimpleNamespace(file_dict={"main.py": "pass"}))

    if is_valid:
        selector._validate_grade_script("print('result')", reference_exp, str(tmp_path))  # noqa: SLF001
    else:
        with pytest.raises(RuntimeError, match="valid JSON object"):
            selector._validate_grade_script("print('result')", reference_exp, str(tmp_path))  # noqa: SLF001


@pytest.mark.offline
def test_incompatible_cached_grade_script_is_regenerated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_folder = tmp_path / "mock"
    (mock_folder / "workspace_input").mkdir(parents=True)
    (mock_folder / "workspace_input" / "label.csv").touch()
    (mock_folder / "data.py").write_text("pass")
    (mock_folder / "grade.py").write_text("print({'score': 0.75})")

    selector = object.__new__(ValidationSelector)
    selector.sample_code_path = tmp_path / "samples"
    selector.sample_rate = 0.8
    validate_mock = Mock(side_effect=RuntimeError("legacy output"))
    monkeypatch.setattr(selector, "_validate_grade_script", validate_mock)
    strict_grade_code = 'import json; print(json.dumps({"score": 0.75}))'
    generate_mock = Mock(return_value=strict_grade_code)
    monkeypatch.setattr(selector, "_generate_and_run_script", generate_mock)
    monkeypatch.setattr(selector, "print_code", Mock())
    reference_exp = SimpleNamespace(experiment_workspace=SimpleNamespace(file_dict={"main.py": "pass"}))

    _, grade_code = selector._prepare_validation_scripts(  # noqa: SLF001
        reference_exp, "example-competition", str(mock_folder),
    )

    assert grade_code == strict_grade_code
    assert (mock_folder / "grade.py").read_text() == strict_grade_code
    generate_mock.assert_called_once()


@pytest.mark.offline
def test_training_metrics_parser_does_not_execute_python(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    stdout = f"Running training\n{{'train_runtime': __import__('pathlib').Path('{marker}').touch()}}"
    validator = object.__new__(LLMConfigValidator)

    parsed = validator._parse_execution_log(stdout, 0)  # noqa: SLF001

    assert str(marker) not in parsed
    assert not marker.exists()
