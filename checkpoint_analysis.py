# # Sourcing from mle_score.txt
# import os
# import re
# import json
# import pickle
# import pandas as pd
# from pathlib import Path
# from rdagent.log.storage import FileStorage
# from rdagent.log.mle_summary import extract_mle_json
# from scripts.exp.researcher.utils import get_valid_submission
# from rdagent.scenarios.data_science.experiment.experiment import DSExperiment

# def extract_checkpoint_info(log_file):
#     m = re.match(r"(.+)_(\d+)\.log", log_file.name)
#     if m:
#         comp = m.group(1)
#         cp = int(m.group(2))
#         return comp, cp
#     return None, None

# def get_loop_result(method, comp, loop_id, baseline_root, researcher_root):
#     if method == "researcher":
#         comp_folder = researcher_root / f"{comp}"
#     else:
#         comp_folder = baseline_root / f"{comp}"
#     loop_folder = comp_folder / f"Loop_{loop_id}"
#     fs = FileStorage(loop_folder)
#     for msg in fs.iter_msg():
#         if "running" in msg.tag and isinstance(msg.content, DSExperiment):
#             submission_path = msg.content.experiment_workspace.workspace_path / "submission.csv"
#             if submission_path.exists():
#                 scores_path = msg.content.experiment_workspace.workspace_path / "scores.csv"
#                 grade_output_path = msg.content.experiment_workspace.workspace_path / "mle_score.txt"
#                 if not grade_output_path.exists():
#                     raise FileNotFoundError(f"mle_score.txt in {grade_output_path} not found, generate it first!")
#                 grade_output = extract_mle_json(grade_output_path.read_text())
#                 if grade_output:
#                     return grade_output.get("score")
#     return None

# def get_loop_idea(loop_path, method, comp):
#     if method == "researcher":
#         fs = FileStorage(loop_path)
#         for msg in fs.iter_msg():
#             if msg.tag and "direct_exp_gen" in msg.tag and hasattr(msg.content, "hypothesis"):
#                 h = msg.content.hypothesis
#                 return f"component: {h.component}, hypothesis: {h.hypothesis}"
#     elif method == "baseline":
#         baseline_folder = Path(f"./log1/{comp}")
#         for msg in FileStorage(baseline_folder).iter_msg():
#             if msg.tag and "direct_exp_gen" in msg.tag and hasattr(msg.content, "hypothesis"):
#                 h = msg.content.hypothesis
#                 task_info = ""
#                 if hasattr(msg.content, "pending_tasks_list"):
#                     task_info = ", tasks: " + str(msg.content.pending_tasks_list)
#                 return f"hypothesis: {h.hypothesis}{task_info}"
#     return ""

# def get_checkpoint_type(comp_folder, checkpoint_id):
#     first_loop, last_loop = get_valid_submission(comp_folder)
#     if first_loop == checkpoint_id:
#         return "early proposal"
#     if last_loop == checkpoint_id and last_loop != first_loop:
#         return "late proposal"
#     return ""

# def process_log_file(log_file, baseline_root, researcher_root):
#     comp, cp = extract_checkpoint_info(log_file)
#     if comp is None:
#         print(f"Skipping log file {log_file.name}: cannot extract info")
#         return []
#     checkpoint_id = cp
#     loop_id = cp + 1
#     print(f"Processing competition '{comp}' with checkpoint {checkpoint_id} -> loop {loop_id}")
#     rows = []
#     for method in ["baseline", "researcher"]:
#         if method == "researcher":
#             comp_folder = researcher_root / f"{comp}"
#         else:
#             comp_folder = baseline_root / f"{comp}"
#         if not comp_folder.exists():
#             print(f"Folder not found: {comp_folder}")
#             continue
#         loop_folder = comp_folder / f"Loop_{loop_id}"
#         if not loop_folder.exists():
#             print(f"Loop folder not found: {loop_folder}")
#             continue
#         ctype = get_checkpoint_type(comp_folder, checkpoint_id)
#         result = get_loop_result(method, comp, loop_id, baseline_root, researcher_root)
#         idea = get_loop_idea(loop_folder, method, comp)
#         print(f"Found {method} data for '{comp}' at Loop_{loop_id}: type {ctype}, score {result}")
#         row = {
#             "method": method,
#             "competition": comp,
#             "checkpoint_id": checkpoint_id,
#             "checkpoint_type": ctype,
#             "results": result,
#             "idea": idea
#         }
#         rows.append(row)
#     return rows

# def visualize_dataframe(df, title="DataFrame Visualization"):
#     import plotly.graph_objects as go
#     fig = go.Figure(data=[go.Table(
#         header=dict(values=list(df.columns), fill_color='paleturquoise', align='left'),
#         cells=dict(values=[df[col].tolist() for col in df.columns], fill_color='lavender', align='left')
#     )])
#     fig.update_layout(title=title)
#     fig.show()

# def main():
#     baseline_root = Path("./log1")
#     researcher_root = Path("./log2")
#     researcher_logs = list(researcher_root.glob("*.log"))
#     print(f"Found {len(researcher_logs)} researcher log files")
#     all_rows = []
#     for log_file in researcher_logs:
#         print(f"Processing log file: {log_file.name}")
#         rows = process_log_file(log_file, baseline_root, researcher_root)
#         all_rows.extend(rows)
#     if not all_rows:
#         print("No data found")
#         return
#     df = pd.DataFrame(all_rows, columns=["method", "competition", "checkpoint_id", "checkpoint_type", "results", "idea"])
#     grp = df.groupby(["idea", "checkpoint_type"]).agg({"results": ["count", "mean"]}).reset_index()
#     print("原始数据 DataFrame:")
#     print(df.to_string(index=False))
#     print("\n分组汇总数据 Groupby DataFrame:")
#     print(grp.to_string(index=False))
#     script_dir = Path(__file__).parent
#     out_file = script_dir / "checkpoint_analysis.csv"
#     with out_file.open("w", encoding="utf-8") as f:
#         f.write("原始数据\n")
#         df.to_csv(f, index=False)
#         f.write("\n\n分组汇总数据\n")
#         grp.to_csv(f, index=False)
#     print(f"Saved analysis results to {out_file}")
#     visualize_dataframe(df, title="原始数据 DataFrame")
#     visualize_dataframe(grp, title="分组汇总数据 Groupby DataFrame")

# if __name__ == "__main__":
#     main()


import re
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from rdagent.log.storage import FileStorage
from rdagent.app.data_science.loop import DataScienceRDLoop

def get_score(loop_folder):
    m = re.match(r"Loop_(\d+)", loop_folder.name)
    if not m:
        return None
    loop_id = m.group(1)
    comp_folder = loop_folder.parent  # competition folder
    session_path = comp_folder / "__session__" / loop_id / "4_record"
    if not session_path.exists():
        return None
    output_path = "log2"
    kaggle_loop = DataScienceRDLoop.load(str(session_path), output_path)
    if kaggle_loop.trace.hist:
        return kaggle_loop.trace.hist[-1][0].result
    return None

def get_idea(loop_folder):
    fs = FileStorage(loop_folder)
    for msg in fs.iter_msg():
        # if msg.tag and "direct_exp_gen" in msg.tag and hasattr(msg.content, "hypothesis"):
        if hasattr(msg.content, "hypothesis"):
            h = msg.content.hypothesis
            return f"component: {h.component}, hypothesis: {h.hypothesis}"
    return None

def visualize_dataframe(df, title="DataFrame Visualization"):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color="paleturquoise", align="left"),
        cells=dict(values=[df[col].tolist() for col in df.columns], fill_color="lavender", align="left")
    )])
    fig.update_layout(title=title)
    fig.show()

def main():
    base_path = Path("/data/userdata/v-xhong/researcher/RD-Agent/")
    folders = [d for d in base_path.iterdir() if d.is_dir() and (d.name.startswith("log_researcher_") or d.name.startswith("log_baseline_"))]
    rows = []
    for folder in folders:
        method = "researcher" if "researcher" in folder.name else "baseline"
        for file in folder.glob("*.log"):
            m = re.match(r"(.+?)_(early|late)_(\d+)\.log", file.name)
            if not m:
                continue
            competition, ctype, chkpt_id = m.group(1), m.group(2), int(m.group(3))
            ctype_full = "early proposal" if ctype == "early" else "late proposal"
            new_loop_folder = folder / competition / f"Loop_{chkpt_id + 1}"
            original_loop_folder = base_path / "log_checkpoint" / competition / f"Loop_{chkpt_id}"
            new_score = get_score(new_loop_folder)
            orig_score = get_score(original_loop_folder)
            metric_increment = (
                (new_score.iloc[:, 0].max() - orig_score.iloc[:, 0].max()) * 100 / orig_score.iloc[:, 0].max()
                if new_score is not None and orig_score is not None and not orig_score.empty and orig_score.iloc[:, 0].max() != 0
                else None
            )
            idea = get_idea(new_loop_folder)
            rows.append({
                "Method": method,
                "Competition": competition,
                "Checkpoint ID": chkpt_id,
                "Checkpoint Type": ctype_full,
                "Metric Increment (%)": metric_increment,
                "Idea": idea
            })
    df = pd.DataFrame(rows)
    output_csv = base_path / "checkpoint_analysis.csv"
    df.to_csv(output_csv, index=False)
    visualize_dataframe(df, title="Checkpoint Analysis")

if __name__ == "__main__":
    main()
