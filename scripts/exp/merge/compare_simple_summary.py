from pathlib import Path
import pandas as pd
from .gen_final_res import select_best

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
