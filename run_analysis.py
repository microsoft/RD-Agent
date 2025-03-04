import argparse
import os
import re
import pandas as pd
from rdagent.app.data_science.loop import DataScienceRDLoop


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True, help="Root path that containing log_checkpoint, log_baseline, log_researcher.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final results.")
    return parser.parse_args()


def extract_exp_path(root_path):
    '''
    Args:
        root_path (str): The path to the root directory containing the `log_<key>` subdirectories.

    Returns:
        log_dict: A dictionary where keys are categories (e.g., `checkpoint`, `baseline`) and values are lists of paths to
              `<competition_name>` directories, formatted appropriately.
    log
    - log_checkpoint
        - <competition_name>
    - log_baseline
        - round_0
            - <competition_name>
    - log_researcher
        - round_0
            - <competition_name>
    '''
    log_dict = {}
    for f in os.scandir(root_path):
        if f.is_dir():
            match = re.match(r"log_(.+)", f.name) # such as log_checkpoint, log_baseline, log_researcher
            if match:
                ckpt_type = match.group(1) # checkpoint, baseline, researcher
                if ckpt_type not in log_dict:
                    log_dict[ckpt_type] = []

                for subdir in os.scandir(f.path):
                    if subdir.is_dir():
                        round_match = re.match(r"round_(\d+)", subdir.name) # such as round_0, round_1
                        if round_match:
                            for competition_dir in os.scandir(subdir.path):
                                if competition_dir.is_dir():
                                    log_dict[ckpt_type].append(f"{subdir.name}/{competition_dir.name}")
                        else:
                            log_dict[ckpt_type].append(subdir.name)
    return log_dict


def get_last_step(session_path):
    steps = os.listdir(session_path)
    idx, step = -1, ""
    for s in steps:
        cur_idx = int(re.findall(r'\d+', s)[0])
        if cur_idx > idx:
            idx = cur_idx
            step = s
    return step


def evaluate_trace(competition_path: str, loop_idx: int):
    session_path = f"{competition_path}/__session__/{loop_idx}"
    session_path = f"{session_path}/{get_last_step(session_path)}"

    kaggle_loop = DataScienceRDLoop.load(session_path)
    try: 
        exp, feedback = kaggle_loop.trace.hist[loop_idx]
        return {"Loop Index": loop_idx, 
                "Score": exp.result.loc['ensemble'].iloc[0] if exp.result is not None else None, 
                "Metric": exp.result.columns[0] if exp.result is not None else None, 
                "Hypothesis": str(exp.hypothesis), 
                "Feedback": str(feedback)}
    except:
        return {"Loop Index": loop_idx, 
                "Score": None, 
                "Metric": None, 
                "Hypothesis": None, 
                "Feedback": None}


def analyze_single_loop(competition_path: str, loop_idx: int):
    loop_result_dict = {}
    loop_result_dict.update(evaluate_trace(competition_path, int(loop_idx)))
    
    return loop_result_dict


def analyze_single_competition(ckpt_type, competition_path):
    result_table = []
    pattern = r"round_(\d+)"
    exp_idx = 0
    match = re.search(pattern, competition_path)
    if match:
        exp_idx = int(match.group(1))
    loop_list = os.listdir(f"{competition_path}/__session__")
    for loop_idx in loop_list:
        comp_result = {
            "Competition": competition_path.split("/")[-1],
            "Exp Index": exp_idx,
            "Type": ckpt_type
        }
        comp_result.update(analyze_single_loop(competition_path, loop_idx))
        result_table.append(comp_result)
    return result_table


def group_result_table(result_table, priority):
    df = pd.DataFrame(result_table)
    df = df.groupby(priority).first().reset_index()
    return df


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


def main():
    args = arg_parser()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    log_dict = extract_exp_path(args.root_path)
    result_table = []

    for key in log_dict:
        for path in log_dict[key]:
            log_path = f"{args.root_path}/log_{key}/{path}"
            print(f"Processing {log_path}")
            result_table.extend(analyze_single_competition(key, log_path))

    result_df = group_result_table(result_table, ["Competition", "Loop Index", "Type", "Exp Index"])
    result_df.to_csv(f"{args.output_path}/results.csv", index=False)

    filtered_df = filter_checkpoint_rows(result_df)
    filtered_df.to_csv(f"{args.output_path}/results_filtered.csv", index=False)

    agg_df = aggregate_results(filtered_df)
    agg_df.to_csv(f"{args.output_path}/aggregated_results.csv", index=False)


if __name__ == "__main__":
    main()

