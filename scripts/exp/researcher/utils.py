import fire, re
from pathlib import Path
from rdagent.log.mle_summary import *
from rdagent.log.storage import FileStorage
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.core.proposal import ExperimentFeedback


def get_valid_submission(log_trace_path):
    log_trace_path = Path(log_trace_path)
    first_loop = last_loop = -1
    for msg in FileStorage(log_trace_path).iter_msg():  # messages in log trace
        try:
            cur_loop = int(re.findall(r'\d+', msg.tag)[0])
            if cur_loop > last_loop:
                last_loop = cur_loop
        except:
            pass
        if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
            if "competition" in msg.tag:
                pass
            if "direct_exp_gen" in msg.tag and isinstance(msg.content, DSExperiment):
                pass
            if "running" in msg.tag and isinstance(msg.content, DSExperiment):
                submission_path = msg.content.experiment_workspace.workspace_path / "submission.csv"
                if submission_path.exists():
                    if first_loop == -1: 
                        first_loop = int(re.findall(r'\d+', msg.tag)[0])
            if "feedback" in msg.tag and "evolving" not in msg.tag and isinstance(msg.content, ExperimentFeedback) and bool(msg.content):
                pass
    return first_loop, last_loop

if __name__ == "__main__":
    fire.Fire(
        {
            "first_valid": get_valid_submission,
        }
    )
