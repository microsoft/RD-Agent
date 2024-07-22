from rdagent.log.ui.web import WebView, QlibFactorTraceWindow
from rdagent.log.storage import FileStorage

WebView(QlibFactorTraceWindow(show_common_logs=False, show_llm=False)).display(FileStorage("./log/2024-07-18_08-37-00-477228"))