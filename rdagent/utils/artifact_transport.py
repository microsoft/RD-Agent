import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ARTIFACT_DUMP_CODE = r"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _dump_safe_artifact(value, root, name):
    if isinstance(value, pd.DataFrame):
        file_name = f"{name}.parquet"
        value.to_parquet(root / file_name)
        return {"type": "dataframe", "file": file_name}
    if isinstance(value, pd.Series):
        file_name = f"{name}.parquet"
        value.to_frame("__rdagent_value__").to_parquet(root / file_name)
        series_name = (
            value.name if value.name is None or isinstance(value.name, (bool, int, float, str)) else str(value.name)
        )
        return {"type": "series", "file": file_name, "name": series_name}
    if isinstance(value, pd.Index):
        file_name = f"{name}.parquet"
        value.to_series(index=range(len(value)), name="__rdagent_value__").to_frame().to_parquet(root / file_name)
        index_name = (
            value.name if value.name is None or isinstance(value.name, (bool, int, float, str)) else str(value.name)
        )
        return {"type": "index", "file": file_name, "name": index_name}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {"type": "ndarray_json", "value": value.tolist()}
        file_name = f"{name}.npy"
        np.save(root / file_name, value, allow_pickle=False)
        return {"type": "ndarray", "file": file_name}
    if isinstance(value, LabelEncoder):
        return {"type": "label_encoder", "classes": value.classes_.tolist()}
    if isinstance(value, np.generic):
        return {"type": "scalar", "value": value.item()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "scalar", "value": value}
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [_dump_safe_artifact(item, root, f"{name}_{index}") for index, item in enumerate(value)],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [
                [_dump_safe_artifact(key, root, f"{name}_key_{index}"),
                 _dump_safe_artifact(item, root, f"{name}_value_{index}")]
                for index, (key, item) in enumerate(value.items())
            ],
        }
    raise TypeError(f"Unsupported result artifact type: {type(value).__module__}.{type(value).__qualname__}")


def dump_safe_artifacts(values, output_folder="rdagent_artifacts"):
    root = Path(output_folder)
    root.mkdir(parents=True, exist_ok=True)
    manifest = [_dump_safe_artifact(value, root, f"artifact_{index}") for index, value in enumerate(values)]
    (root / "manifest.json").write_text(json.dumps(manifest))
"""


def _artifact_path(root: Path, file_name: str) -> Path:
    path = (root / file_name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        message = f"Artifact file escapes bundle directory: {file_name}"
        raise ValueError(message) from exc
    return path


def _load_node(node: dict[str, Any], root: Path) -> Any:  # noqa: PLR0911
    artifact_type = node["type"]
    if artifact_type == "dataframe":
        return pd.read_parquet(_artifact_path(root, node["file"]))
    if artifact_type == "series":
        series = pd.read_parquet(_artifact_path(root, node["file"]))["__rdagent_value__"]
        series.name = node.get("name")
        return series
    if artifact_type == "index":
        values = pd.read_parquet(_artifact_path(root, node["file"]))["__rdagent_value__"]
        return pd.Index(values, name=node.get("name"))
    if artifact_type == "ndarray":
        return np.load(_artifact_path(root, node["file"]), allow_pickle=False)
    if artifact_type == "ndarray_json":
        return np.asarray(node["value"])
    if artifact_type == "label_encoder":
        encoder = LabelEncoder()
        encoder.classes_ = np.asarray(node["classes"])
        return encoder
    if artifact_type == "scalar":
        return node.get("value")
    if artifact_type in {"list", "tuple"}:
        values = [_load_node(item, root) for item in node["items"]]
        return tuple(values) if artifact_type == "tuple" else values
    if artifact_type == "dict":
        return {_load_node(key, root): _load_node(value, root) for key, value in node["items"]}
    message = f"Unsupported result artifact type: {artifact_type}"
    raise ValueError(message)


def load_artifact_bundle(manifest_path: str | Path) -> list[Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, list):
        message = "Artifact manifest must contain a list"
        raise TypeError(message)
    return [_load_node(node, path.parent) for node in manifest]


def load_result_artifact(path: str | Path) -> list[Any]:
    artifact_path = Path(path)
    if artifact_path.name == "manifest.json" and artifact_path.parent.name == "rdagent_artifacts":
        return load_artifact_bundle(artifact_path)
    if artifact_path.suffix == ".json":
        return [json.loads(artifact_path.read_text())]
    if artifact_path.suffix == ".txt":
        return [artifact_path.read_text()]
    if artifact_path.suffix == ".npy":
        return [np.load(artifact_path, allow_pickle=False)]
    if artifact_path.suffix == ".parquet":
        return [pd.read_parquet(artifact_path)]
    message = f"Unsafe result artifact format: {artifact_path.suffix}"
    raise ValueError(message)
