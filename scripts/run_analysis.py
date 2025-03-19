import argparse
import os
import re
import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.core.proposal import HypothesisFeedback
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.utils.env import DockerEnv, MLEBDockerConf
# from scripts.exp.researcher.utils import extract_JSON


import json
import re


mle_de_conf = MLEBDockerConf()
mle_de_conf.extra_volumes = {
    f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data",
}
de = DockerEnv(conf=mle_de_conf)
de.prepare()

def extract_JSON(text):
    # Try to directly load the text as JSON
    try:
        extracted_text = json.loads(text)
        if isinstance(extracted_text, dict):
            return [extracted_text]
        elif isinstance(extracted_text, list):
            return extracted_text
        else:
            return []
    except json.JSONDecodeError:
        pass
   
    # Try to extract the outermost JSON array
    try:
        match = re.search(r'\[(?:[^\[\]]|\[.*\])*\]', text)
        if match:
            extracted_text = json.loads(match.group(0))
            return extracted_text
    except json.JSONDecodeError:
        pass
   
    # Try to extract the first JSON object
    try:
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            extracted_text = json.loads(match.group(0))
            return [extracted_text]
    except json.JSONDecodeError:
        pass

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True, help="Root path that containing log_checkpoint, log_baseline, log_researcher.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final results.")
    return parser.parse_args()


def extract_exp_path(root_path):
    """
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
    """
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


def get_session_path(competition_path: str, loop_idx: str):
    session_path = f"{competition_path}/__session__/{loop_idx}"
    last_step = get_last_step(session_path)
    session_path = f"{session_path}/{last_step}"
    return session_path, last_step


def get_score(competition: str, exp: DSExperiment):
    metric, validation_score, test_score = None, None, None
    # get validation score
    if exp.result is not None: 
        metric = exp.result.columns[0] if exp.result is not None else None
        validation_score = float(exp.result.loc['ensemble'].iloc[0]) if exp.result is not None else None

    # get test score
    submission_path = exp.experiment_workspace.workspace_path / "submission.csv"
    test_score_path = exp.experiment_workspace.workspace_path / "mle_score.txt"
    if submission_path.exists():
        if not test_score_path.exists():
            exp.experiment_workspace.execute(
                                env=de,
                                entry=f"mlebench grade-sample submission.csv {competition} --data-dir /mle/data > mle_score.txt 2>&1",
                            )
            exp.experiment_workspace.execute(env=de, entry="chmod 777 mle_score.txt")
        test_score = extract_JSON(test_score_path.read_text())[0]['score']
    return metric, validation_score, test_score


def get_analysis(exp: DSExperiment, fb: HypothesisFeedback):
    idea, feature = None, None
    try: 
        idea = exp.hypothesis.ideas
        feature = ""
        features = exp.hypothesis.features
        for i, feat in enumerate(features):
            feature += f"{feat['label']} {i}: {feat['feature']}"
    except:
        pass
    hypothesis = str(exp.hypothesis)
    feedback = str(fb)
    decision = fb.decision

    return idea, feature, hypothesis, feedback, decision


def analyze_single_loop(competition_path: str, loop_idx: int):
    # load loop from session
    loop_result_dict = {"Loop Index": int(loop_idx), 
                        "Metric": None, 
                        "Validation Score": None, 
                        "Test Score": None, 
                        "Hypothesis": None, 
                        "Idea": None, 
                        "Feature": None, 
                        "Feedback": None, 
                        "Decision": None}
    session_path, last_step = get_session_path(competition_path, loop_idx)
    if last_step == "4_record": 
        ds_loop = DataScienceRDLoop.load(path=session_path, do_truncate=False)
        competition = ds_loop.trace.scen.competition
        exp, fb = ds_loop.trace.hist[-1]

        metric, validation_score, test_score = get_score(competition, exp)
        loop_result_dict['Metric'] = metric
        loop_result_dict['Validation Score'] = validation_score
        loop_result_dict['Test Score'] = test_score

        idea, feature, hypothesis, feedback, decision = get_analysis(exp, fb)
        loop_result_dict['Idea'] = idea
        loop_result_dict['Feature'] = feature
        loop_result_dict['Hypothesis'] = hypothesis
        loop_result_dict['Feedback']= feedback
        loop_result_dict['Decision'] = decision
    
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
    grouped = df.groupby(["Competition", "Loop Index", "Type", "Metric"], as_index=False)
    agg_df = grouped.agg(
        Validation_Score=('Validation Score', lambda x: x.dropna().mean()),
        Test_Score=('Test Score', lambda x: x.dropna().mean()),
        Success_Rate=('Test Score', lambda x: x.notna().sum() / len(x))
    )

    agg_df = agg_df[["Competition", "Loop Index", "Type", "Validation_Score", "Test_Score", "Success_Rate"]]
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
    result_df.to_markdown(f"{args.output_path}/results.md", index=False)

    filtered_df = filter_checkpoint_rows(result_df)
    filtered_df.to_csv(f"{args.output_path}/results_filtered.csv", index=False)

    agg_df = aggregate_results(filtered_df)
    agg_df.to_csv(f"{args.output_path}/aggregated_results.csv", index=False)


if __name__ == "__main__":
    main()
