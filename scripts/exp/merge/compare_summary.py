from itertools import combinations

# exp_list = [
#     "deciding-cod",
#     "exotic-frog",  # 12h version
#     "civil-reindeer",
#     "selected-worm",  # o3-4.1 | 12h version
#     "prepared-salmon",  # o1-4.1
# ]
#
# exp_submit_order = [
#     "civil-reindeer_prepared-salmon",
#     "deciding-cod_civil-reindeer",
#     "deciding-cod_prepared-salmon",
#     "exotic-frog_civil-reindeer",
#     "exotic-frog_selected-worm",
#     "civil-reindeer_selected-worm",
#     "deciding-cod_exotic-frog",
#     "deciding-cod_selected-worm",
#     "exotic-frog_prepared-salmon",
#     "selected-worm_prepared-salmon",
# ]
# exp_submit_name = [
#     "measured-jackal",
#     "wondrous-bluegill",
#     "optimum-sole",
#     "noted-whale",
#     "great-guppy",
#     "inviting-quail",
#     "amusing-glider",
#     "charmed-pheasant",
#     "big-yak",
#     "factual-trout",
# ]
#
# finished_exp = [
#     "measured-jackal",
#     "wondrous-bluegill",
#     "optimum-sole",
#     "noted-whale",
#     "great-guppy",
#     # "inviting-quail",
#     "amusing-glider",
#     "charmed-pheasant",
#     "big-yak",
#     "factual-trout",
# ]

exp_list = [
    "civil-reindeer",
    "selected-worm",  # o3-4.1 | 12h version
    "prepared-salmon",  # o1-4.1
]

exp_submit_order = [
    "civil-reindeer_prepared-salmon",
    "civil-reindeer_selected-worm",
    "selected-worm_prepared-salmon",
]

exp_submit_name = [
    "internal-kid",
    "renewing-jackal",
    "sacred-wildcat",
]

finished_exp = [
    "internal-kid",
    "renewing-jackal",
    "sacred-wildcat",
]


def get_exp_name(exp1, exp2):
    """
    Return the merged experiment submit name given two experiment keys.
    If the combination is not found, return the joined name.
    Only works if the order matches exactly with exp_submit_order.
    """
    key = f"{exp1}_{exp2}"
    if key in exp_submit_order:
        idx = exp_submit_order.index(key)
        return exp_submit_name[idx]
    return key  # fallback: just join with underscore


# for exp1, exp2 in combinations(exp_list, 2):
#     exp_name = get_exp_name(exp1, exp2)
#     print(f"Exp1: {exp1}, Exp2: {exp2}, Merged Name: {exp_name}")
#     if exp_name in finished_exp:
#         break

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
    base13h = pd.read_hdf("13h_base_df.h5", "data")
    base13h_exp1 = base13h[base13h.index.str.contains(exp1)]
    base13h_exp2 = base13h[base13h.index.str.contains(exp2)]

    basefull = pd.read_hdf("full_base_df.h5", "data")
    basefull_exp1 = basefull[basefull.index.str.contains(exp1)]
    basefull_exp2 = basefull[basefull.index.str.contains(exp2)]

    merge = pd.read_hdf("merge_base_df.h5", "data")
    merge_exp = merge[merge.index.str.contains(exp_name)]

    # baseline
    baseline = pd.concat([base13h_exp1, base13h_exp2])
    # ours
    ours = pd.concat([base12h_exp1, base12h_exp2, merge_exp])

    console.print(Rule("[bold magenta]For Verification[/bold magenta]", style="magenta"))
    basefull_exp1_stat = select_best(basefull_exp1)
    basefull_exp2_stat = select_best(basefull_exp2)
    merge_exp_stat = select_best(merge_exp)

    print(merge_exp_stat)
    print(basefull_exp1_stat)
    print(basefull_exp2_stat)


    # baseline 1: full trace
    console.print(Rule("[bold magenta]Baseline 1: Full Trace[/bold magenta]", style="magenta"))
    basefull_exp1_stat = select_best(basefull_exp1)
    basefull_exp2_stat = select_best(basefull_exp2)
    baseline_stat = select_best(baseline)
    print(basefull_exp1_stat)
    print(basefull_exp2_stat)

    # baseline 2: merging
    console.print(Rule("[bold magenta]Baseline 2: Simple Merging[/bold magenta]", style="magenta"))
    base13h_exp1_stat = select_best(base13h_exp1)
    base13h_exp2_stat = select_best(base13h_exp2)
    baseline_stat = select_best(baseline)
    print(base13h_exp1_stat)
    print(base13h_exp2_stat)
    print(baseline_stat)


# Ours
    console.print(Rule("[bold magenta]Ours: Code Mergeing[/bold magenta]", style="magenta"))
    base12h_exp1_stat = select_best(base12h_exp1)
    base12h_exp2_stat = select_best(base12h_exp2)
    ours_stat = select_best(ours)
    print(base12h_exp1_stat)
    print(base12h_exp2_stat)
    print(ours_stat)

    ours_w_sel, _ = select_best(ours, return_sel=True)
    ours_w_sel[ours_w_sel.index.str.contains(exp_name)]
    ours_w_sel["SOTA Exp Score (valid)"]

    # def ana_stat(gdf):
    #     direction = 1 if (gdf["Gold Threshold"] >= gdf["Medium Threshold"]).all() else -1
    #     if not ours_w_sel[ours_w_sel.index.str.contains(exp_name)].empty:
    #         break
    #
    # for key, gdf in ours_w_sel.groupby("Competition"):
    #     direction = 1 if (gdf["Gold Threshold"] >= gdf["Medium Threshold"]).all() else -1
    #     if not gdf[gdf.index.str.contains(exp_name)].empty:
    #         gdf[gdf.index.str.contains(exp_name)]
    #         break


for exp1, exp2 in combinations(exp_list, 2):
    exp_name = get_exp_name(exp1, exp2)
    console.print(Rule(f"[bold green]Exp1: {exp1}, Exp2: {exp2}, Merged Name: {exp_name}[/bold green]", style="green"))
    print()
    print_comparision(exp1, exp2, exp_name)


# ----------------------------------------

# for exp_name in finished_exp:
#     merge_exp = merge[merge.index.str.contains(exp_name)]
#     print(select_best(merge_exp))
