from rdagent.log.ui.web import WebView, QlibFactorTraceWindow
from rdagent.log.storage import FileStorage

WebView(QlibFactorTraceWindow(show_common_logs=True, show_llm=True)).display(FileStorage("./log/2024-07-18_08-37-00-477228"))