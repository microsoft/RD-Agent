import pickle
import streamlit as st
from pathlib import Path
from rdagent.log.base import Storage, View
from rdagent.log.storage import FileStorage
from rdagent.log.base import Message
from datetime import timezone, datetime
from collections import defaultdict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
    from rdagent.core.proposal import Hypothesis, HypothesisFeedback
    from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
    from rdagent.components.coder.factor_coder.factor import FactorTask, FactorFBWorkspace
    from rdagent.components.coder.factor_coder.CoSTEER.evaluators import FactorSingleFeedback

class ProcessView(View):
    def __init__(self, trace_path: Path):
        # Save logs to your desired data structure
        # ...
        pass

    def display(s: Storage, watch: bool = False):
        pass


class WebView(View):
    r"""

    We have tree structure for sequence

    session
    |      \
    ... defined by user ...
    |                              \
    info1 -> info2 -> ... -> info3 -> ...  overtime.

    <message dispature>
          |  | -  dispatch according to uri(e.g. `a.b.c. ...`)
    Frontend is composed of windows.
    Each window can individually display the message flow.

    Some design principles:
        session1.module(e.g. implement).
        `s.log(a.b.1.c) s.log(a.b.2.c)` should not handed over to users.

    An display example:

        W1 write factor
        W2 evaluate factor
        W3 backtest

        W123
            R
            RX
            RXX
            RX

        W4
            trace r1 r2 r3 r4

    What to do next?
    1. Data structure
    2. Map path like `a.b.c` to frontend components
    3. Display logic
    """

    def __init__(self, ui: 'StWindow'):
        self.ui = ui
        # Save logs to your desired data structure
        # ...

    def display(self, s: Storage, watch: bool = False):

        for msg in s.iter_msg():  # iterate overtime
            # NOTE:  iter_msg will correctly seperate the information.
            # TODO: msg may support streaming mode.
            self.ui.consume_msg(msg)



# TODO: Implement the following classes
class StWindow:

    def __init__(self, container: 'DeltaGenerator'):
        self.container = container

    def consume_msg(self, msg: Message):
        msg_str = f"{msg.timestamp.astimezone(timezone.utc).isoformat()} | {msg.level} | {msg.caller} - {msg.content}"
        self.container.write(msg_str)


class LLMWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator', session_name: str="common"):
        self.container = container
        self.container.subheader(f"{session_name} Messages")

    def consume_msg(self, msg: Message):
        self.container.chat_message('User').write(f"{msg.content}")


class CodeWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator'):
        self.container = container.empty()
    
    def consume_msg(self, msg: Message):
        self.container.code(msg.content, language="python")


class MultiProcessWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator', inner_class: str = "STLWindow"):
        '''
        inner_class: STLWindow 子类名称, 用来实例化多进程窗口的内部窗口实例
        '''
        self.container = container.empty()
        self.tabs_cache = defaultdict(list)
        self.inner_class = eval(inner_class)
        
    
    def consume_msg(self, msg: Message):
        name = msg.pid_trace.split("-")[-1]
        self.tabs_cache[name].append(msg)

        tabs = self.container.tabs(list(self.tabs_cache.keys()))
        for i, name in enumerate(self.tabs_cache):
            inner_win: StWindow = self.inner_class(tabs[i])
            for m in self.tabs_cache[name]:
                inner_win.consume_msg(m)


class HypothesisRelatedWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator'):
        self.container = container
    
    def consume_msg(self, msg: Message):
        h: Hypothesis | HypothesisFeedback = msg.content
        self.container.text(str(h))


class QlibFactorUI(StWindow):

    def __init__(self, container: 'DeltaGenerator' = st.container()):
        super().__init__(container)
        self.pid_level = 0
        self.tag_level = 0
        self.current_win = container.container()

    def consume_msg(self, msg: Message):
        pid_level = msg.pid_trace.count("-")
        tag_level = msg.tag.count(".")

        if pid_level > self.pid_level:
            self.pid_level = pid_level
            self.current_win = MultiProcessWindow(st.container(), "STLWindow")
        
        if tag_level > self.tag_level:
            self.tag_level = tag_level


        self.current_win.consume_msg(msg)


if __name__ == "__main__":
    WebView(QlibFactorUI()).display(FileStorage("./log/test_trace"))