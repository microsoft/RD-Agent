import pytest

from rdagent.scenarios.data_science.debug.data import JsonReducer


@pytest.mark.offline
def test_json_reducer_extracts_filename_from_unnamed_key() -> None:
    assert JsonReducer().extract_filename({"id": "img_001.jpg"}) == "img_001.jpg"
