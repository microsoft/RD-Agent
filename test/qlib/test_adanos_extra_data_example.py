import ast
import builtins
import io
import json
import random
import socket
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from docutils import nodes
from docutils.core import publish_doctree
from jinja2 import Environment, StrictUndefined

pytestmark = pytest.mark.offline
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/scens/data_agent_fin.rst"
COLUMNS = ["$adanos_sentiment", "$adanos_mentions", "$adanos_buzz"]


def documented_block(name: str) -> str:
    document = publish_doctree(
        DOC.read_text(encoding="utf-8"),
        settings_overrides={"halt_level": 6, "report_level": 5, "warning_stream": io.StringIO()},
    )
    blocks = [block.astext() for block in document.findall(nodes.literal_block) if name in block.get("names", [])]
    assert len(blocks) == 1, f"Expected one documented code block named {name}"
    return blocks[0]


@pytest.fixture(autouse=True)
def offline_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("This example must not use a network, provider, model, or runtime environment")

    original_import = builtins.__import__

    def offline_import(name, *args, **kwargs):
        if name.startswith(("rdagent.oai", "qlib", "docker", "openai", "litellm")):
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(builtins, "__import__", offline_import)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def recipe() -> dict:
    namespace: dict = {}
    exec(compile(documented_block("adanos-offline-recipe"), str(DOC), "exec"), namespace)
    return namespace


@pytest.fixture
def report() -> dict:
    return {
        "ticker": "AAPL",
        "found": True,
        "daily_trend": [
            {"date": "2026-05-06", "mentions": 4, "sentiment_score": 0.5, "buzz_score": 20.0},
            {"date": "2026-05-04", "mentions": 0, "sentiment_score": None, "buzz_score": None},
        ],
    }


def isolated_native_file_description(path: Path) -> str:
    native_path = ROOT / "rdagent/scenarios/qlib/experiment/utils.py"
    tree = ast.parse(native_path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_file_desc")
    native = {"pd": pd, "Path": Path, "random": random, "Environment": Environment, "StrictUndefined": StrictUndefined}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(native_path), "exec"), native)
    with pd.option_context("display.max_columns", 20, "display.max_rows", 60, "display.max_colwidth", 50):
        return native["get_file_desc"](path)


def test_saved_json_to_native_extra_hdf(recipe: dict, report: dict, tmp_path: Path) -> None:
    saved = json.loads(documented_block("adanos-offline-report"))
    assert saved == report
    (tmp_path / "aapl-report.json").write_text(json.dumps(saved), encoding="utf-8")
    (tmp_path / "instrument-map.json").write_text(json.dumps({"AAPL": "AAPL"}), encoding="utf-8")
    protected = {}
    for folder in (tmp_path / "research/full", tmp_path / "research/debug"):
        folder.mkdir(parents=True)
        for filename in ("daily_pv.h5", "result.h5", "README.md"):
            path = folder / filename
            protected[path] = b"existing research input"
            path.write_bytes(protected[path])

    exec(compile(documented_block("adanos-offline-usage"), str(DOC), "exec"), recipe)
    expected = pd.DataFrame(
        [[float("nan"), 0.0, float("nan")], [0.5, 4.0, 20.0]],
        columns=COLUMNS,
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-05-04"), "AAPL"), (pd.Timestamp("2026-05-06"), "AAPL")],
            names=["datetime", "instrument"],
        ),
    )
    for folder in (tmp_path / "research/full", tmp_path / "research/debug"):
        output = folder / "adanos_reddit.h5"
        with pd.HDFStore(output, mode="r") as store:
            assert store.keys() == ["/data"]
            pd.testing.assert_frame_equal(store["data"], expected)
        assert sorted(path.name for path in folder.iterdir()) == [
            "README.md",
            "adanos_reddit.h5",
            "daily_pv.h5",
            "result.h5",
        ]
    assert all(path.read_bytes() == content for path, content in protected.items())
    assert not (tmp_path / "git_ignore_folder").exists()

    description = isolated_native_file_description(output)
    assert "HDF5 Data File" in description
    assert "['datetime', 'instrument']" in description
    assert all(f"{column}: float64" in description for column in COLUMNS)
    (tmp_path / "native-description.txt").write_text(description, encoding="utf-8")


def test_explicit_mapping_and_multiple_reports(recipe: dict, report: dict) -> None:
    other = deepcopy(report)
    other["ticker"] = "MSFT"
    other["daily_trend"] = [other["daily_trend"][0]]
    frame = recipe["to_adanos_frame"]([other, report], {"AAPL": "US_AAPL", "MSFT": "US_MSFT"})
    assert list(frame.index) == [
        (pd.Timestamp("2026-05-04"), "US_AAPL"),
        (pd.Timestamp("2026-05-06"), "US_AAPL"),
        (pd.Timestamp("2026-05-06"), "US_MSFT"),
    ]
    assert frame.loc[(pd.Timestamp("2026-05-06"), "US_MSFT")].tolist() == [0.5, 4.0, 20.0]


@pytest.mark.parametrize("trend", [None, [], "absent"])
@pytest.mark.parametrize("found", [True, False])
def test_no_observations_stay_empty(recipe: dict, trend, found: bool) -> None:
    report = {"ticker": "AAPL", "found": found, "daily_trend": trend}
    if trend == "absent":
        del report["daily_trend"]
    frame = recipe["to_adanos_frame"]([report], {"AAPL": "AAPL"})
    assert frame.empty
    assert frame.index.names == ["datetime", "instrument"]
    assert list(frame.columns) == COLUMNS
    assert all(dtype == "float64" for dtype in frame.dtypes)


@pytest.mark.parametrize(
    "payload",
    [
        {"detail": "Unauthorized"},
        {"detail": {"error": "Historical data limit exceeded"}},
        {"error": "rate limited", "ticker": "AAPL"},
        {"ticker": "AAPL", "daily_trend": {}},
        {"ticker": "AAPL", "daily_trend": [None]},
        {"ticker": "AAPL", "found": "false"},
        [],
    ],
)
def test_rejects_saved_errors_and_bad_reports(recipe: dict, payload) -> None:
    with pytest.raises(ValueError):
        recipe["to_adanos_frame"]([payload], {"AAPL": "AAPL"})


@pytest.mark.parametrize("mapping", [{}, {"AAPL": ""}, {"AAPL": " AAPL"}, {"AAPL": 1}, {"AAPL": "X", "MSFT": "X"}])
def test_rejects_unknown_or_ambiguous_identity(recipe: dict, report: dict, mapping: dict) -> None:
    with pytest.raises(ValueError):
        recipe["to_adanos_frame"]([report], mapping)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("date", "2026-02-30"),
        ("date", "20260506"),
        ("date", "2026-05-06T00:00:00Z"),
        ("date", None),
        ("sentiment_score", 1.1),
        ("sentiment_score", -1.1),
        ("sentiment_score", float("nan")),
        ("buzz_score", float("inf")),
        ("buzz_score", 101),
        ("buzz_score", -1),
        ("buzz_score", "20"),
        ("mentions", -1),
        ("mentions", 1.5),
        ("mentions", True),
        ("mentions", None),
    ],
)
def test_rejects_invalid_observations(recipe: dict, report: dict, field: str, value) -> None:
    report["daily_trend"][0][field] = value
    with pytest.raises(ValueError):
        recipe["to_adanos_frame"]([report], {"AAPL": "AAPL"})


def test_rejects_duplicate_and_contradictory_observations(recipe: dict, report: dict) -> None:
    with pytest.raises(ValueError):
        recipe["to_adanos_frame"]([report, report], {"AAPL": "AAPL"})
    report["found"] = False
    with pytest.raises(ValueError):
        recipe["to_adanos_frame"]([report], {"AAPL": "AAPL"})


def test_exclusive_write_and_missing_folder(recipe: dict, report: dict, tmp_path: Path) -> None:
    frame = recipe["to_adanos_frame"]([report], {"AAPL": "AAPL"})
    output = tmp_path / "adanos_reddit.h5"
    output.write_bytes(b"keep existing data")
    with pytest.raises(FileExistsError):
        recipe["write_adanos_extra"](frame, tmp_path)
    assert output.read_bytes() == b"keep existing data"
    output.unlink()
    output.symlink_to(tmp_path / "absent-target")
    with pytest.raises(FileExistsError):
        recipe["write_adanos_extra"](frame, tmp_path)
    assert output.is_symlink()
    assert not output.exists()
    with pytest.raises(FileNotFoundError):
        recipe["write_adanos_extra"](frame, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_failed_hdf_write_leaves_no_output(recipe: dict, report: dict, tmp_path: Path, monkeypatch) -> None:
    frame = recipe["to_adanos_frame"]([report], {"AAPL": "AAPL"})

    def fail(*args, **kwargs):
        raise OSError("Synthetic disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_hdf", fail)
    with pytest.raises(OSError):
        recipe["write_adanos_extra"](frame, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_destination_created_during_write_is_not_replaced(
    recipe: dict, report: dict, tmp_path: Path, monkeypatch
) -> None:
    frame = recipe["to_adanos_frame"]([report], {"AAPL": "AAPL"})
    real_link = recipe["os"].link

    def competing_writer(source, destination):
        destination.write_bytes(b"another writer's snapshot")
        real_link(source, destination)

    monkeypatch.setattr(recipe["os"], "link", competing_writer)
    with pytest.raises(FileExistsError):
        recipe["write_adanos_extra"](frame, tmp_path)
    assert (tmp_path / "adanos_reddit.h5").read_bytes() == b"another writer's snapshot"
    assert [path.name for path in tmp_path.iterdir()] == ["adanos_reddit.h5"]


def test_empty_frame_hdf_has_no_imputed_rows(recipe: dict, tmp_path: Path) -> None:
    frame = recipe["to_adanos_frame"]([], {})
    output = recipe["write_adanos_extra"](frame, tmp_path)
    restored = pd.read_hdf(output, key="data")
    pd.testing.assert_frame_equal(restored, frame)
