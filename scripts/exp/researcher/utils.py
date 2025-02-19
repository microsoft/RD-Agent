import fire, re
from pathlib import Path
from rdagent.log.mle_summary import *
from rdagent.log.storage import FileStorage
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.core.proposal import ExperimentFeedback


def get_first_valid_submission(log_trace_path):
    log_trace_path = Path(log_trace_path)
    for msg in FileStorage(log_trace_path).iter_msg():  # messages in log trace
        if msg.tag and "llm" not in msg.tag and "session" not in msg.tag:
            if "competition" in msg.tag:
                pass
            if "direct_exp_gen" in msg.tag and isinstance(msg.content, DSExperiment):
                pass
            if "running" in msg.tag and isinstance(msg.content, DSExperiment):
                submission_path = msg.content.experiment_workspace.workspace_path / "submission.csv"
                if submission_path.exists():
                    loop = int(re.findall(r'\d+', msg.tag)[0])
                    return loop 
            if "feedback" in msg.tag and "evolving" not in msg.tag and isinstance(msg.content, ExperimentFeedback) and bool(msg.content):
                pass
    
    loop = int(re.findall(r'\d+', msg.tag)[0])
    return loop

if __name__ == "__main__":
    fire.Fire(
        {
            "first_valid": get_first_valid_submission,
        }
    )
