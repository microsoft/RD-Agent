import pandas as pd
import pytest

from rdagent.app.finetune.llm.ui import ft_summary
from rdagent.app.rl.ui import rl_summary


@pytest.mark.offline
@pytest.mark.parametrize("module", [ft_summary, rl_summary])
def test_style_df_with_decisions_styles_every_cell(module) -> None:
    # `styles.iloc[row][col] = ...` assigns into a temporary row copy under pandas 3
    # Copy-on-Write, which left every cell unstyled.
    df = pd.DataFrame({"Loop 0": ["OK", "X"], "Loop 1": ["C", "R"]}, index=["job-a", "job-b"])
    decisions = pd.DataFrame({"Loop 0": [True, False], "Loop 1": [None, None]}, index=df.index)

    ctx = module.style_df_with_decisions(df, decisions)._compute().ctx

    for row_idx in range(len(df)):
        for col_idx in range(len(df.columns)):
            assert ctx[(row_idx, col_idx)], (row_idx, col_idx)
