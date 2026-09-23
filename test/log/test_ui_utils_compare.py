import pandas as pd
import pytest

from rdagent.log.ui import utils


@pytest.mark.offline
def test_compare_select_best_keeps_best_row_per_competition(monkeypatch, tmp_path) -> None:
    # pandas 3 removes the grouping column from each group passed to
    # groupby().apply(), so reading cdf["Competition"] there raised KeyError.
    base_df = pd.DataFrame(
        {
            "Competition": ["comp-a", "comp-a", "comp-b", "comp-b"],
            "SOTA Exp Score (valid, to_submit)": [0.1, 0.9, 0.7, 0.2],
        },
        index=["a-1", "a-2", "b-1", "b-2"],
    )
    summary = {idx: {} for idx in base_df.index}
    saved = []
    monkeypatch.setattr(utils, "get_summary_df", lambda log_folders, hours: (summary, base_df))
    monkeypatch.setattr(utils, "get_metric_direction", lambda competition: competition == "comp-a")
    monkeypatch.setattr(pd.DataFrame, "to_hdf", lambda self, *args, **kwargs: saved.append(self))

    utils.compare(exp_list=["exp"], output=str(tmp_path / "out.h5"), hours=None, select_best=True)

    # comp-a: higher is better -> a-2; comp-b: lower is better -> b-2
    assert sorted(saved[0].index) == ["a-2", "b-2"]
