import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from rdagent.utils.artifact_transport import (
    ARTIFACT_DUMP_CODE,
    load_artifact_bundle,
    load_result_artifact,
)


@pytest.mark.offline
def test_safe_artifact_bundle_round_trip(tmp_path: Path) -> None:
    namespace: dict = {}
    exec(ARTIFACT_DUMP_CODE, namespace)  # noqa: S102
    frame = pd.DataFrame({"feature": [1.0, 2.0]}, index=[10, 20])
    series = pd.Series([3, 4], name="target", index=[10, 20])
    array = np.asarray([[1, 2], [3, 4]])
    encoder = LabelEncoder().fit(["a", "b"])
    values = [frame, series, array, [encoder, "id"]]

    namespace["dump_safe_artifacts"](values, tmp_path / "rdagent_artifacts")
    restored = load_artifact_bundle(tmp_path / "rdagent_artifacts" / "manifest.json")

    pd.testing.assert_frame_equal(restored[0], frame)
    pd.testing.assert_series_equal(restored[1], series)
    np.testing.assert_array_equal(restored[2], array)
    np.testing.assert_array_equal(restored[3][0].classes_, encoder.classes_)
    assert restored[3][1] == "id"


@pytest.mark.offline
def test_artifact_bundle_rejects_file_path_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "rdagent_artifacts"
    bundle.mkdir()
    manifest = [{"type": "ndarray", "file": "../outside.npy"}]
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="escapes"):
        load_artifact_bundle(bundle / "manifest.json")


@pytest.mark.offline
def test_result_loader_rejects_pickle(tmp_path: Path) -> None:
    result_path = tmp_path / "result.pkl"
    result_path.write_bytes(b"not deserialized")

    with pytest.raises(ValueError, match="Unsafe result artifact format"):
        load_result_artifact(result_path)
