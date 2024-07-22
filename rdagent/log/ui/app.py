from rdagent.log.ui.web import WebView, QlibFactorTraceWindow, ProposalTraceWindow, mock_msg
from rdagent.log.storage import FileStorage, Message

# WebView(QlibFactorTraceWindow(show_common_logs=True, show_llm=True)).display(FileStorage("./log/2024-07-18_08-37-00-477228"))


from pathlib import Path
import pickle
with Path('./log/progress.pkl').open('rb') as f:
    obj = pickle.load(f)
ProposalTraceWindow().consume_msg(mock_msg(obj[0]))