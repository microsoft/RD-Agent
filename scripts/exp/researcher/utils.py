import re, os
from rdagent.app.data_science.loop import DataScienceRDLoop

def get_loop_idx(log_trace_path):
    session_path = f"{log_trace_path}/__session__"
    es_loop = ls_loop = -1
    for loop in os.listdir(session_path):
        loop_idx = int(loop)
        session = f"{session_path}/{loop}"
        session = f"{session}/{get_last_step(session)}"
        kaggle_loop = DataScienceRDLoop.load(path=session)
        if kaggle_loop.trace.next_incomplete_component() is None: # all component are complete
            if loop_idx < es_loop or es_loop == -1:
                es_loop = loop_idx
            
        if loop_idx > ls_loop:
            ls_loop = loop_idx

    return es_loop, ls_loop


def get_last_step(session_path):
    steps = os.listdir(session_path)
    idx, step = -1, ""
    for s in steps:
        cur_idx = int(re.findall(r'\d+', s)[0])
        if cur_idx > idx:
            idx = cur_idx
            step = s
    return step


from pathlib import Path
import pickle
class Saver:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def dump(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with path.open("rb") as f:
            return pickle.load(f)