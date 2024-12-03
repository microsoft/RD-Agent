import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from rdagent.log.storage import FileStorage
from rdagent.scenarios.kaggle.kaggle_crawler import (
    leaderboard_scores,
)
import pandas as pd

def collect_results(log_path) -> list[dict]:
    summary = []
    log_storage = FileStorage(Path(log_path))
    evaluation_metric_direction = None
    # Extract score from trace using the same approach as UI
    for msg in log_storage.iter_msg():
        if "scenario" in msg.tag:
            competition_name = msg.content.competition # Find the competition name     
            leaderboard = leaderboard_scores(competition_name)
            evaluation_metric_direction = float(leaderboard[0]) > float(leaderboard[-1])
 
        if "runner result" in msg.tag:
            if msg.content.result is not None:
                score = msg.content.result
                summary.append({
                    "competition_name": competition_name,
                    "score": score,
                    "workspace": msg.content.experiment_workspace.workspace_path,
                    "evaluation_metric_direction": evaluation_metric_direction
                })
    return summary

def generate_summary(results, output_path):
    summary = {
        "configs": {}, #TODO: add config? 
        "best_result": {"competition_name": None, "score": None},
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        #Add other metrics that we want to track in the future (eg. is there successive increase?)
    }
    for result in results:
        # Update best result
        if result["evaluation_metric_direction"]:
            if (result["score"] is not None and 
                (summary["best_result"]["score"] is None or 
                (result["score"].iloc[0] > summary["best_result"]["score"]))):
                summary["best_result"].update({
                    "score": result["score"].iloc[0] if isinstance(result["score"], pd.Series) else result["score"],
                    "competition_name": result["competition_name"]
                })
        else:
            if (result["score"] is not None and 
                (summary["best_result"]["score"] is None or 
                (result["score"].iloc[0] < summary["best_result"]["score"]))):
                summary["best_result"].update({
                    "score": result["score"].iloc[0] if isinstance(result["score"], pd.Series) else result["score"],
                    "competition_name": result["competition_name"]
                })
    
    # Convert Series to scalar or list if necessary
    for key, value in summary.items():
        if isinstance(value, pd.Series):
            summary[key] = value.tolist()  # Convert Series to list
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, pd.Series):
                    value[sub_key] = sub_value.tolist()  # Convert Series to list

    with open(output_path, "w") as f: 
        json.dump(summary, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='Collect and summarize experiment results')
    parser.add_argument('--log_path', type=str, required=True,
                       help='Path to the log directory containing experiment results')
    parser.add_argument('--output_name', type=str, default='summary.json',
                       help='Name of the output summary file (default: summary.json)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    log_path = Path(args.log_path)
    
    # Verify the log path exists
    if not log_path.exists():
        raise FileNotFoundError(f"Log path does not exist: {log_path}")
    
    results = collect_results(log_path)
    output_path = log_path / args.output_name
    generate_summary(results, output_path)
    print("Summary generated successfully at", output_path)

