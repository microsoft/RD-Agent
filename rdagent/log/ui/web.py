import streamlit as st
from pathlib import Path
from rdagent.log.base import Storage, View
from rdagent.log.storage import FileStorage
from rdagent.log.base import Message
from datetime import timezone, datetime
from collections import defaultdict

from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
    from rdagent.core.proposal import Hypothesis, HypothesisFeedback
    from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
    from rdagent.components.coder.factor_coder.factor import FactorTask, FactorFBWorkspace
    from rdagent.components.coder.factor_coder.CoSTEER.evaluators import FactorSingleFeedback

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
        self.container = container.expander(f"{session_name} Messages")

    def consume_msg(self, msg: Message):
        role = 'user' if 'Role:user' in msg.content else 'assistant'
        self.container.chat_message(role).write(f"{msg.content.split('Content: ')[1]}")


class CodeWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator'):
        self.container = container.empty()
    
    def consume_msg(self, msg: Message):
        self.container.code(msg.content, language="python")


class TabsWindow(StWindow):
    '''
    For windows with stream messages, will refresh when a new tab is created.
    '''
    def __init__(self,
                 container: 'DeltaGenerator',
                 inner_class: str = "STLWindow",
                 mapper: Callable[[Message], str] = lambda x: x.pid_trace):
        
        self.inner_class = eval(inner_class)
        self.mapper = mapper

        self.container = container.empty()
        self.tab_windows: dict[str, StWindow] = defaultdict(None)
        self.tab_caches: dict[str, list[Message]] = defaultdict(list)
        
    
    def consume_msg(self, msg: Message):
        name = self.mapper(msg)

        if name not in self.tab_windows:
            # new tab need to be created, current streamlit container need to be updated.
            names = list(self.tab_windows.keys()) + [name]

            if len(names) == 1:
                tabs = [self.container.container()]
            else:
                tabs = self.container.tabs(names)

            for id, name in enumerate(names):
                self.tab_windows[name] = self.inner_class(tabs[id])
            
            # consume the cache
            for name in self.tab_caches:
                for msg in self.tab_caches[name]:
                    self.tab_windows[name].consume_msg(msg)
        
        self.tab_caches[name].append(msg)
        self.tab_windows[name].consume_msg(msg)



class HypothesisRelatedWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator'):
        self.container = container
    
    def consume_msg(self, msg: Message):
        h: Hypothesis | HypothesisFeedback = msg.content
        self.container.text(str(h))


class FactorTaskWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator'):
        self.container = container
    
    def consume_msg(self, msg: Message):
        task: FactorTask = msg.content
        self.container.text(str(task))


class FactorFeedbackWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator'):
        self.container = container
    
    def consume_msg(self, msg: Message):
        fb: FactorSingleFeedback = msg.content
        self.container.text(str(fb))


class FactorWorkspaceWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator'):
        self.container = container
    
    def consume_msg(self, msg: Message):
        ws: FactorFBWorkspace = msg.content
        self.container.text(str(ws))


class QlibFactorExpWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator'):
        self.container = container
    
    def consume_msg(self, msg: Message):
        exp: QlibFactorExperiment = msg.content
        self.container.text(str(exp))


class QlibFactorTraceWindow(StWindow):

    def __init__(self, container: 'DeltaGenerator' = st.container()):
        super().__init__(container)
        self.pid_trace = ''
        self.current_tag = ''
        self.current_win = StWindow(self.container.container())

    def consume_msg(self, msg: Message):
        if len(msg.tag) > len(self.current_tag):
            
            # write a header about current task, if it is llm message, not write.
            if not msg.tag.endswith('llm_messages'):
                self.container.subheader(msg.tag.replace('.', ' > '), divider=True)
            
            self.current_tag = msg.tag

            if msg.tag.endswith('llm_messages'):
                self.current_win = LLMWindow(self.container.container())
            elif isinstance(msg.content, Hypothesis) or isinstance(msg.content, HypothesisFeedback):
                self.current_win = HypothesisRelatedWindow(self.container.container())
            elif isinstance(msg.content, FactorTask):
                self.current_win = FactorTaskWindow(self.container.container())
            elif isinstance(msg.content, FactorFBWorkspace):
                self.current_win = FactorWorkspaceWindow(self.container.container())
            elif isinstance(msg.content, FactorSingleFeedback):
                self.current_win = FactorFeedbackWindow(self.container.container())
            elif isinstance(msg.content, QlibFactorExperiment):
                self.current_win = QlibFactorExpWindow(self.container.container())
            else:
                self.current_win = StWindow(self.container)

        elif len(msg.tag) < len(self.current_tag):
            # write a divider when the task is finished
            self.container.markdown("---")

        self.current_win.consume_msg(msg)


if __name__ == "__main__":
    WebView(QlibFactorTraceWindow()).display(FileStorage("./log/2024-07-18_08-37-00-477228"))
