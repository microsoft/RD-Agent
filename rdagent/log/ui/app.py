from rdagent.log.ui.web import WebView, QlibTraceWindow, TraceObjWindow, mock_msg
from rdagent.log.storage import FileStorage, Message
from rdagent.core.proposal import Trace
from pathlib import Path
import pickle


# show logs folder
# WebView(QlibTraceWindow(show_common_logs=False, show_llm=False)).display(FileStorage("./log/2024-07-22_03-01-12-021659"))


# load Trace obj
with Path('./log/step_trace.pkl').open('rb') as f:
    obj = pickle.load(f)
    trace: Trace = obj[-1]

# show Trace obj
# TraceObjWindow().consume_msg(mock_msg(trace))
