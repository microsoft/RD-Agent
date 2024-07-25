from rdagent.log.ui.web import WebView, SimpleTraceWindow, TraceObjWindow, mock_msg, TraceWindow
from rdagent.log.storage import FileStorage, Message
from rdagent.core.proposal import Trace
from pathlib import Path
import pickle


# show logs folder
WebView(TraceWindow()).display(FileStorage("/data/home/bowen/workspace/RD-Agent/log/yuante/2024-07-24_04-03-33-691119"))
# WebView(TraceWindow()).display(FileStorage("/data/home/bowen/workspace/RD-Agent/log/2024-07-22_03-01-12-021659"))
# WebView(TraceWindow()).display(FileStorage("./log/2024-07-18_08-37-00-477228"))


# load Trace obj
# with Path('./log/step_trace.pkl').open('rb') as f:
#     obj = pickle.load(f)
#     trace: Trace = obj[-1]

# show Trace obj
# TraceObjWindow().consume_msg(mock_msg(trace))
