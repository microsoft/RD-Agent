import os
import json
from pathlib import Path
from datetime import datetime

def collect_results(dir_path) -> list[dict]:
    summary = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith("_result.json"):
                config_name = file.replace("_result.json", "")
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    # Extract both CV and Kaggle submission results
                    summary.append({
                        "config": config_name,
                        "cv_results": data.get("cv_score", None),
                        "kaggle_score": data.get("kaggle_score", None),
                        "trace": data.get("trace", {})
                    })
    return summary

def generate_summary(results, output_path):
    summary = {
        "configs": {},
        "best_cv_result": {"config": None, "score": None},
        "best_kaggle_result": {"config": None, "score": None},
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    for result in results:
        config = result["config"]
        metrics = {
            "cv_score": result["cv_results"],
            "kaggle_score": result["kaggle_score"],
            "iterations": len(result["trace"].get("steps", [])),
            "best_model": result["trace"].get("best_model")
        }
        
        summary["configs"][config] = metrics
        
        # Update best CV result
        if (metrics["cv_score"] is not None and 
            (summary["best_cv_result"]["score"] is None or 
             metrics["cv_score"] > summary["best_cv_result"]["score"])):
            summary["best_cv_result"].update({
                "config": config,
                "score": metrics["cv_score"]
            })
            
        # Update best Kaggle result
        if (metrics["kaggle_score"] is not None and 
            (summary["best_kaggle_result"]["score"] is None or 
             metrics["kaggle_score"] > summary["best_kaggle_result"]["score"])):
            summary["best_kaggle_result"].update({
                "config": config,
                "score": metrics["kaggle_score"]
            })
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    result_dir = os.path.join(os.getenv("EXP_DIR"), "results")
    results = collect_results(result_dir)
    generate_summary(results, os.path.join(result_dir, "summary.json"))
    print("Summary generated successfully at ", os.path.join(result_dir, "summary.json"))

