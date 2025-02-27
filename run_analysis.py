import argparse
import os
import re
import pandas as pd
from rdagent.app.data_science.loop import DataScienceRDLoop

def group_result_table(result_table, priority):
    df = pd.DataFrame(result_table)
    df = df.groupby(priority, as_index=False).agg({"Score": "mean"})
    df = df.sort_values(by=priority, ascending=[True] * len(priority))
    return df

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
    if kaggle_loop.trace.hist:
        return kaggle_loop.trace.hist[-1][0].result
    return None

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
    ckpt_type, exp_idx = "checkpoint", 0
    match = re.search(pattern, log_path)
    if match:
        ckpt_type = match.group(1)
        exp_idx = match.group(2)
    competitions = [entry for entry in sorted(os.scandir(log_path), key=lambda e: e.name) if entry.is_dir()]
    for c in competitions:
        competition_path = f"{log_path}/{c.name}"
        try:
            loop_list = os.listdir(f"{competition_path}/__session__")
        except:
            continue
        for loop_idx in loop_list:
            comp_result = {
                "Competition": competition_path.split("/")[-1],
                "Index": exp_idx,
                "Type": ckpt_type
            }
            comp_result.update(analyze_single_competition(competition_path, loop_idx))
            result_table.append(comp_result)
    return result_table

def filter_checkpoint_rows(df):
    df = df.copy()
    df['Loop Index'] = df['Loop Index'].astype(int)
    valid_indices = df[df['Type'].isin(['baseline', 'researcher'])].copy()
    valid_indices['valid'] = valid_indices['Loop Index'] - 1
    valid_map = valid_indices.groupby('Competition')['valid'].agg(lambda x: set(x)).to_dict()
    def keep_row(row):
        if row['Type'] in ['baseline', 'researcher']:
            return True
        elif row['Type'] == 'checkpoint':
            comp = row['Competition']
            valid_set = valid_map.get(comp, set())
            # Drop checkpoint rows if no corresponding baseline/researcher exists
            if not valid_set:
                return False
            return row['Loop Index'] in valid_set
        return False
    return df[df.apply(keep_row, axis=1)]

def aggregate_results(df):
    grouped = df.groupby(["Competition", "Loop Index", "Type"], as_index=False)
    agg_df = grouped['Score'].agg(
        average_score=lambda x: x.dropna().mean(),
        success_rate=lambda x: x.notna().sum() / len(x)
    )
    agg_df['Loop Index'] = agg_df['Loop Index'].astype(int)
    # Build a mapping for checkpoint rows: key = (Competition, Loop Index)
    checkpoint_map = agg_df[agg_df['Type'] == 'checkpoint'].set_index(['Competition', 'Loop Index'])['average_score'].to_dict()
    def calc_increment(row):
        if row['Type'] in ['baseline', 'researcher']:
            comp = row['Competition']
            li = row['Loop Index']
            ref = checkpoint_map.get((comp, li - 1))
            if ref is not None:
                return row['average_score'] - ref
        return None
    agg_df['score increment'] = agg_df.apply(calc_increment, axis=1)
    # Reorder columns as: Competition, Loop Index, Type, average_score, score increment, success_rate
    agg_df = agg_df[["Competition", "Loop Index", "Type", "average_score", "score increment", "success_rate"]]
    return agg_df

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="/data/userdata/v-xuminrui/RD-Agent", help="Path containing log_checkpoint, log_baseline, log_researcher.")
    return parser.parse_args()

def main():
    args = arg_parser()
    result_table = []
    log_dict = {"checkpoint": [], "baseline": [], "researcher": []}
    for f in os.scandir(args.log_path):
        if f.is_dir():
            for key in log_dict:
                if key in f.name:
                    log_dict[key].append(f.name)
    for key in log_dict:
        for name in log_dict[key]:
            log_path = f"{args.log_path}/{name}"
            print(f"Processing {log_path}")
            result_table.extend(analyze_single_folder(log_path))
    result_df = group_result_table(result_table, ["Competition", "Loop Index", "Type", "Index"])
    result_df.to_excel("results.xlsx", index=False)
    filtered_df = filter_checkpoint_rows(result_df)
    filtered_df.to_excel("results_filtered.xlsx", index=False)
    agg_df = aggregate_results(filtered_df)
    agg_df.to_excel("aggregated_results.xlsx", index=False)

if __name__ == "__main__":
    main()

