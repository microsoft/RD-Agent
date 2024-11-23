import os
import json
from pathlib import Path
from datetime import datetime
from rdagent.log.storage import FileStorage

def collect_results(dir_path) -> list[dict]:
    summary = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith("_result.json"):
                config_name = file.replace("_result.json", "")
                log_storage = FileStorage(Path(root))
                
                score = None
                # Extract score from trace using the same approach as UI
                for msg in log_storage.iter_msg():
                    if "runner result" in msg.tag:
                        if msg.content.result is not None:
                            score = msg.content.result
                            break
                
                summary.append({
                    "config": config_name,
                    "score": score,
                    "workspace": str(root)
                })
    return summary

def generate_summary(results, output_path):
    summary = {
        "configs": {},
        "best_result": {"config": None, "score": None},
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    for result in results:
        config = result["config"]
        metrics = {
            "score": result["score"],
            "workspace": result["workspace"]
        }
        
        summary["configs"][config] = metrics
        
        # Update best result
        if (result["score"] is not None and 
            (summary["best_result"]["score"] is None or 
             result["score"] > summary["best_result"]["score"])):
            summary["best_result"].update({
                "config": config,
                "score": result["score"]
            })
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    result_dir = os.path.join(os.getenv("EXP_DIR"), "results")
    results = collect_results(result_dir)
    generate_summary(results, os.path.join(result_dir, "summary.json"))
    print("Summary generated successfully at ", os.path.join(result_dir, "summary.json"))

