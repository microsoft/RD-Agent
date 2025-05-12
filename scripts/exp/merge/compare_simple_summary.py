from pathlib import Path
import pandas as pd
from rdagent.scenarios.kaggle.kaggle_crawler import leaderboard_scores


def get_metric_direction(competition: str):
    leaderboard = leaderboard_scores(competition)
    return float(leaderboard[0]) > float(leaderboard[-1])


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


def num2percent(num: int, total: int, show_origin=True) -> str:
    num = int(num)
    total = int(total)
    if show_origin:
        return f"{num} ({round(num / total * 100, 2)}%)"
    return f"{round(num / total * 100, 2)}%"


def percent_df(df: pd.DataFrame, show_origin=True) -> pd.DataFrame:
    base_df = df.copy(deep=True)

    # Convert columns to object dtype so we can store strings like "14 (53.85%)" without warnings
    columns_to_convert = [
        "Successful Final Decision",
        "Made Submission",
        "Valid Submission",
        "Above Median",
        "Bronze",
        "Silver",
        "Gold",
        "Any Medal",
    ]
    base_df[columns_to_convert] = base_df[columns_to_convert].astype(object)

    for k in base_df.index:
        loop_num = int(base_df.loc[k, "Total Loops"])
        if loop_num != 0:
            base_df.loc[k, "Successful Final Decision"] = num2percent(
                base_df.loc[k, "Successful Final Decision"], loop_num, show_origin
            )
            if base_df.loc[k, "Made Submission"] != 0:
                base_df.loc[k, "V/M"] = (
                    f"{round(base_df.loc[k, 'Valid Submission'] / base_df.loc[k, 'Made Submission'] * 100, 2)}%"
                )
            else:
                base_df.loc[k, "V/M"] = "N/A"
            base_df.loc[k, "Made Submission"] = num2percent(base_df.loc[k, "Made Submission"], loop_num, show_origin)
            base_df.loc[k, "Valid Submission"] = num2percent(base_df.loc[k, "Valid Submission"], loop_num, show_origin)
            base_df.loc[k, "Above Median"] = num2percent(base_df.loc[k, "Above Median"], loop_num, show_origin)
            base_df.loc[k, "Bronze"] = num2percent(base_df.loc[k, "Bronze"], loop_num, show_origin)
            base_df.loc[k, "Silver"] = num2percent(base_df.loc[k, "Silver"], loop_num, show_origin)
            base_df.loc[k, "Gold"] = num2percent(base_df.loc[k, "Gold"], loop_num, show_origin)
            base_df.loc[k, "Any Medal"] = num2percent(base_df.loc[k, "Any Medal"], loop_num, show_origin)

    return base_df


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
    best_idxs = base_df.groupby("Competition").apply(apply_func)
    base_df["Select"] = base_df.index.isin(best_idxs.values)

    base_df = base_df[base_df["Select"]]
    print(f"**统计的比赛数目: :red[{base_df.shape[0]}]**")
    total_stat = (
        base_df[
            [
                "Made Submission",
                "Valid Submission",
                "Above Median",
                "Bronze",
                "Silver",
                "Gold",
                "Any Medal",
            ]
        ]
        != "0 (0.0%)"
    ).sum()
    total_stat.name = "总体统计(%)"
    total_stat.loc["Bronze"] = base_df["Best Result"].value_counts().get("bronze", 0)
    total_stat.loc["Silver"] = base_df["Best Result"].value_counts().get("silver", 0)
    total_stat.loc["Gold"] = base_df["Best Result"].value_counts().get("gold", 0)
    total_stat = total_stat / base_df.shape[0] * 100

    # SOTA Exp 统计
    se_counts = base_df["SOTA Exp"].value_counts(dropna=True)
    se_counts.loc["made_submission"] = se_counts.sum()
    se_counts.loc["Any Medal"] = se_counts.get("gold", 0) + se_counts.get("silver", 0) + se_counts.get("bronze", 0)
    se_counts.loc["above_median"] = se_counts.get("above_median", 0) + se_counts.get("Any Medal", 0)
    se_counts.loc["valid_submission"] = se_counts.get("valid_submission", 0) + se_counts.get("above_median", 0)

    sota_exp_stat = pd.Series(index=total_stat.index, dtype=int, name="SOTA Exp 统计(%)")
    sota_exp_stat.loc["Made Submission"] = se_counts.get("made_submission", 0)
    sota_exp_stat.loc["Valid Submission"] = se_counts.get("valid_submission", 0)
    sota_exp_stat.loc["Above Median"] = se_counts.get("above_median", 0)
    sota_exp_stat.loc["Bronze"] = se_counts.get("bronze", 0)
    sota_exp_stat.loc["Silver"] = se_counts.get("silver", 0)
    sota_exp_stat.loc["Gold"] = se_counts.get("gold", 0)
    sota_exp_stat.loc["Any Medal"] = se_counts.get("Any Medal", 0)
    sota_exp_stat = sota_exp_stat / base_df.shape[0] * 100

    stat_df = pd.concat([total_stat, sota_exp_stat], axis=1)
    if return_sel:
        return base_df, stat_df
    return stat_df


# for verification
# use rich package with colored divider to print  "for verification"
from rich.console import Console
from rich.rule import Rule

console = Console()
def print_comparision(exp1, exp2, exp_name):
    base12h = pd.read_hdf("12h_base_df.h5", "data")
    base12h_exp1 = base12h[base12h.index.str.contains(exp1)]
    base12h_exp2 = base12h[base12h.index.str.contains(exp2)]

    basefull = pd.read_hdf("full_base_df.h5", "data")
    basefull_exp1 = basefull[basefull.index.str.contains(exp1)]
    basefull_exp2 = basefull[basefull.index.str.contains(exp2)]


    # baseline 1: full trace
    console.print(Rule("[bold magenta]Baseline 1: Full Trace[/bold magenta]", style="magenta"))
    basefull_exp1_stat = select_best(basefull_exp1)
    basefull_exp2_stat = select_best(basefull_exp2)
    baseline_stat = select_best(baseline)
    print(basefull_exp1_stat)
    print(basefull_exp2_stat)

    # baseline 2: merging
    baseline = pd.concat([base12h_exp1, base12h_exp2])
    console.print(Rule("[bold magenta]Baseline 2: Simple Merging[/bold magenta]", style="magenta"))
    base12h_exp1_stat = select_best(base12h_exp1)
    base12h_exp2_stat = select_best(base12h_exp2)
    baseline_stat = select_best(baseline)
    print(base12h_exp1_stat)
    print(base12h_exp2_stat)
    print(baseline_stat)

from itertools import combinations
names = []
for p in Path("mle-bench-res").glob("*.h5"):
    names.append(p.stem)


from tqdm import tqdm
all_stat = {}
for exp1, exp2 in tqdm(combinations(names, 2)):
    exp1_12h = pd.read_hdf(f"mle-bench-res/{exp1}.h5", "data")
    exp2_12h = pd.read_hdf(f"mle-bench-res/{exp2}.h5", "data")
    exp1_full = pd.read_hdf(f"mle-bench-res-full/{exp1}.h5", "data")
    exp2_full = pd.read_hdf(f"mle-bench-res-full/{exp2}.h5", "data")
    merge_12h = pd.concat([exp1_12h, exp2_12h])
    col = "SOTA Exp 统计(%)"
    stat = {
        "exp1_full": select_best(exp1_full)[col],
        "exp2_full": select_best(exp2_full)[col],
        "exp1_12h": select_best(exp1_12h)[col],
        "exp2_12h": select_best(exp2_12h)[col],
        "merge_12h": select_best(merge_12h)[col],
    }
    all_stat[f"{exp1}_{exp2}"] = pd.DataFrame(stat)

all_stat_df = pd.concat(all_stat)

all_stat_df.to_hdf("all_stat.h5", "data")

# pd.DataFrame(stat)
