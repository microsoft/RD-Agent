import os
import json

def collect_results(dir_path) -> list[dict]:
    summary = []
    for root, _, filies in os.walk(dir_path):
        for file in filies:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    summary.append(data)
    return summary

def generate_summary(results, output_path):
    # First analyze the results and generate a summary
    # For each experiment, we find the best result, the metric, and result trajectory
    #TODO: Implement this 

    # Then write the summary to the output path
    with open(output_path, "w") as f:
        json.dump(results, f, indent = 4)

if __name__ == "__main__":
    result_dir = os.path.join(os.getenv("EXP_DIR"), "results")
    results = collect_results(result_dir)
    generate_summary(results, os.path.join(result_dir, "summary.json"))
    print("Summary generated successfully at ", os.path.join(result_dir, "summary.json"))

