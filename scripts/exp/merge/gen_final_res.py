from pathlib import Path
import pandas as pd
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction
from rdagent.log.ui.utils import percent_df, get_statistics_df


def remove_busy_owl(df):
    """remove busy-owl competition; becuase it is a fixed competition that contains successful results"""
    if df.shape[0] > 75:
        df = df[~df.index.str.contains("^busy-owl - seti-breakthrough-listen|^busy-owl - whale-categorization-playground|^uncommon-macaque - seti-breakthrough-listen|^uncommon-macaque - whale-categorization-playground")]
    return df


def select_best(base_df: pd.DataFrame, return_sel: bool=False):
    base_df = remove_busy_owl(base_df.copy())
    base_df = base_df[~base_df.index.duplicated(keep="first")]

    base_df = percent_df(base_df)
    base_df.insert(0, "Select", True)
    def apply_func(cdf: pd.DataFrame):
        cp = cdf["Competition"].values[0]
        md = get_metric_direction(cp)
        # If SOTA Exp Score (valid) column is empty, return the first index
        if cdf["SOTA Exp Score (valid)"].dropna().empty:
            return cdf.index[0]
        if md:
            best_idx = cdf["SOTA Exp Score (valid)"].idxmax()
        else:
            best_idx = cdf["SOTA Exp Score (valid)"].idxmin()
        return best_idx
    best_idxs = base_df.groupby("Competition").apply(apply_func)
    base_df["Select"] = base_df.index.isin(best_idxs.values)

    base_df = base_df[base_df["Select"]]
    print(f"**统计的比赛数目: :red[{base_df.shape[0]}]**")
    
    stat_df = get_statistics_df(base_df)
    if return_sel:
        return base_df, stat_df
    return stat_df

from rdagent.log.ui.utils import ALL, MEDIUM, HIGH, LITE
from pathlib import Path
DIRNAME = Path(__file__).absolute().resolve().parent
col = "SOTA Exp 统计(%)"
res = {}
# for f in Path("mle-bench-res-final").glob("*.pkl"):
# res_path = "./mle-bench-res-final-o3/"  # this version is used on github
res_path = "./mle-bench-res-o3-merge/"

for f in Path(res_path).glob("*.h5"):
    df = pd.read_hdf(f, "data")
    print(f"exp_name: {f.stem}")
    # stat = select_best(df)
    for name, cl in ("ALL", ALL), ("MEDIUM", MEDIUM), ("HIGH", HIGH), ("LITE", LITE):
        df_set = df[df["Competition"].isin(cl)]
        # assert len(df_set) == len(cl)
        res[(f.stem, name)] = select_best(df_set)[col]
res = pd.DataFrame(res)


for case in ["ALL", "MEDIUM", "HIGH", "LITE"]:
    print(f"\n==== {case} ====")
    print(res.loc(axis=1)[:, case].droplevel(axis=1, level=1).T.sort_index().applymap(lambda x: f"{x:.2f}").to_markdown())

# aggregate the "Any Medal" row across different Vs and get mean and std for each V
any_medal = res.loc["Any Medal"].unstack()
print("\n==== Any Medal mean and std for each V ====")
mean = any_medal.mean()
std = any_medal.std()
summary = pd.DataFrame({"mean": mean, "std": std})
print(summary.T.to_markdown())
# Format the table to show mean ± std in one cell, like
# | Agent | Low == Lite (%) | Medium (%) | High (%) | All (%) |
# | AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |
# Now, 'sum_df' contains mean and std for each metric (columns: ALL, MEDIUM, HIGH, LITE) across all experiments


def format_mean_std(mean, std, digits=1):
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

# Compute mean and std for each column across the experiments for "Any Medal"
any_medal = res.loc["Any Medal"].unstack()
mean = any_medal.mean()
std = any_medal.std()
summary = pd.DataFrame({"mean": mean, "std": std})

# Format the table to show mean ± std in one cell, like:
# | Agent | Lite (%) | Medium (%) | High (%) | All (%) |
# | AIDE o1-preview | 34.3 ± 2.4 | 8.8 ± 1.1 | 10.0 ± 1.9 | 16.9 ± 1.1 |

header = ["Agent", "Lite (%)", "Medium (%)", "High (%)", "All (%)"]
col_map = {'LITE': 'Lite (%)', 'MEDIUM': 'Medium (%)', 'HIGH': 'High (%)', 'ALL': 'All (%)'}
cols = ['LITE', 'MEDIUM', 'HIGH', 'ALL']  # Define the order to match the header
# Prepare the single summary row
table_rows = []
row_cells = ["Mean ± Std"]
for col in cols:
    if col in summary.index:
        val = format_mean_std(summary.loc[col, "mean"], summary.loc[col, "std"], digits=1)
    else:
        val = "-"
    row_cells.append(val)
table_rows.append(row_cells)

print("\n==== Any Medal mean ± std for each V ====")
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"]*len(header)) + "|")
for row_cells in table_rows:
    print("| " + " | ".join(row_cells) + " |")


# sum_df = pd.concat({"mean": res.groupby(axis=1, level=1).mean(), "std": res.groupby(axis=1, level=1).std()}, axis=1)
# sum_df.swaplevel(0, 1, axis=1).sort_index(axis=1, level=1).sort_index(axis=1, level=0)  # put mean before std

from IPython import embed; embed()
