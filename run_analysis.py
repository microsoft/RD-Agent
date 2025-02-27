import argparse
import os
import re
import pandas as pd
from rdagent.app.data_science.loop import DataScienceRDLoop


def group_result_table(result_table, priority):
    """
    Groups and aggregates the result table based on the specified priority rules.

    Args:
        result_table (list of dict): The input data as a list of dictionaries.
        priority (list of str): The grouping hierarchy (e.g., ["Competition", "Type", "Index", "Loop Index"]).

    Returns:
        pd.DataFrame: The grouped and aggregated DataFrame.
    """
    result_df = pd.DataFrame(result_table)

    grouped_df = result_df.groupby(
        priority,
        as_index=False
    ).agg({
        "Score": "mean"
    })

    grouped_df = grouped_df.sort_values(
        by=priority,
        ascending=[True] * len(priority)
    )

    return grouped_df


def get_last_step(session_path):
    steps = os.listdir(session_path)
    idx, step = -1, ""
    for s in steps:
        cur_idx = int(re.findall(r'\d+', s)[0])
        if cur_idx > idx:
            idx = cur_idx
            step = s
    return step


def get_mle_score(session_path):
    kaggle_loop = DataScienceRDLoop.load(session_path)
    return kaggle_loop.trace.hist[-1][0].result


def analyze_single_competition(competition_path, loop_idx):
    session_path = f"{competition_path}/__session__/{loop_idx}"
    session_path = f"{session_path}/{get_last_step(session_path)}"
    mle_score = get_mle_score(session_path)
    try:
        mle_score = mle_score.iloc[:, 0].mean()
    except:
        mle_score = None
    return {"Loop Index": loop_idx, "Score": mle_score}


def analyze_single_folder(log_path):
    result_table = []
    pattern = r"log_([^_]+)_(\d+)"
    match = re.search(pattern, log_path)
    if match:
        ckpt_type = match.group(1)
        exp_idx = match.group(2)

    competitions = [entry for entry in sorted(os.scandir(log_path), key=lambda e: e.name) if entry.is_dir()]
    for c in competitions:
        competition_path = f"{log_path}/{c.name}"
        try: 
            loop_list = os.listdir(f"{competition_path}/__session__")
            for loop_idx in loop_list:
                competition_result = {
                    "Competition": competition_path.split("/")[-1], 
                    "Index": exp_idx,
                    "Type": ckpt_type
                }
                competition_result.update(analyze_single_competition(competition_path, loop_idx))
                result_table.append(competition_result)
        except:
            continue
    
    return result_table


def main():
    args = arg_parser()
    result_table = []
    log_dict = {
        "checkpoint": [],
        "baseline": [],
        "researcher": []
    }
    for f in os.scandir(args.log_path):
        if f.is_dir(): 
            for key in log_dict:
                if key in f.name:
                    log_dict[key].append(f.name)
    for key in log_dict:
        for f in log_dict[key]:
            log_path = f"{args.log_path}/{f}"
            print(f"Processing {log_path}")
            result_table.extend(analyze_single_folder(log_path))

    result_df = group_result_table(result_table, ["Competition", "Loop Index", "Type", "Index"])
    result_df.to_csv("results.csv", index=False)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="/data/userdata/v-xuminrui/RD-Agent", help="The path that contains log_checkpoint, log_baseline, log_researcher.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()