import pandas as pd
import streamlit as st
import plotly.express as px

from rdagent.log.base import Storage, View
from rdagent.log.base import Message
from datetime import timezone, datetime
from collections import defaultdict
from copy import deepcopy

from typing import Callable, Type
from streamlit.delta_generator import DeltaGenerator
from rdagent.core.proposal import Hypothesis, HypothesisFeedback

from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

from rdagent.components.coder.factor_coder.factor import FactorTask, FactorFBWorkspace
from rdagent.components.coder.factor_coder.CoSTEER.evaluators import FactorSingleFeedback
from rdagent.components.coder.model_coder.CoSTEER.evaluators import ModelCoderFeedback
from rdagent.components.coder.model_coder.model import ModelTask, ModelFBWorkspace



st.set_page_config(layout="wide")

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



class StWindow:

    def __init__(self, container: 'DeltaGenerator'):
        self.container = container

    def consume_msg(self, msg: Message):
        msg_str = f"{msg.timestamp.astimezone(timezone.utc).isoformat()} | {msg.level} | {msg.caller} - {msg.content}"
        self.container.code(msg_str, language="log")


class LLMWindow(StWindow):
    def __init__(self, container: 'DeltaGenerator', session_name: str="common"):
        self.session_name = session_name
        self.container = container.expander(f"{self.session_name} message")

    def consume_msg(self, msg: Message):
        self.container.chat_message('user').markdown(f"{msg.content}")


class CodeWindow(StWindow):

    def consume_msg(self, msg: Message):
        self.container.code(msg.content, language="python")


class ProgressTabsWindow(StWindow):
    '''
    For windows with stream messages, will refresh when a new tab is created.
    '''
    def __init__(self,
                 container: 'DeltaGenerator',
                 inner_class: Type[StWindow] = StWindow,
                 mapper: Callable[[Message], str] = lambda x: x.pid_trace):
        
        self.inner_class = inner_class
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


class ObjectsTabsWindow(StWindow):
    def __init__(self,
                 container: 'DeltaGenerator',
                 inner_class: Type[StWindow] = StWindow,
                 mapper: Callable[[object], str] = lambda x: str(x),
                 tab_names: list[str] | None = None):
        self.inner_class = inner_class
        self.mapper = mapper
        self.container = container
        self.tab_names = tab_names
    
    def consume_msg(self, msg: Message):
        if isinstance(msg.content, list):
            if self.tab_names:
                assert len(self.tab_names) == len(msg.content), "List of objects should have the same length as provided tab names."
                objs_dict = {self.tab_names[id]: obj for id, obj in enumerate(msg.content)}
            else:
                objs_dict = {self.mapper(obj): obj for obj in msg.content}
        elif not isinstance(msg.content, dict):
            raise ValueError("Message content should be a list or a dict of objects.")
        
        # two many tabs may cause display problem
        tab_names = list(objs_dict.keys())
        tabs = []
        for i in range(0, len(tab_names), 10):
            tabs.extend(self.container.tabs(tab_names[i:i+10]))
        
        for id, obj in enumerate(objs_dict.values()):
            splited_msg = Message(tag=msg.tag,
                                    level=msg.level,
                                    timestamp=msg.timestamp,
                                    caller=msg.caller,
                                    pid_trace=msg.pid_trace,
                                    content=obj)
            self.inner_class(tabs[id]).consume_msg(splited_msg)


class HypothesisWindow(StWindow):
    
    def consume_msg(self, msg: Message):
        h: Hypothesis = msg.content
        self.container.subheader('Hypothesis')
        self.container.markdown(f"""
- **Hypothesis**: {h.hypothesis}
- **Reason**: {h.reason}""")


class HypothesisFeedbackWindow(StWindow):

    def consume_msg(self, msg: Message):
        h: HypothesisFeedback = msg.content
        self.container.subheader('Hypothesis Feedback')
        self.container.markdown(f"""
- **Observations**: {h.observations}
- **Hypothesis Evaluation**: {h.hypothesis_evaluation}
- **New Hypothesis**: {h.new_hypothesis}
- **Decision**: {h.decision}
- **Reason**: {h.reason}""")


class FactorTaskWindow(StWindow):

    def consume_msg(self, msg: Message):
        ft: FactorTask = msg.content

        self.container.markdown(f"**Factor Name**: {ft.factor_name}")
        self.container.markdown(f"**Description**: {ft.factor_description}")
        self.container.latex(f"Formulation: {ft.factor_formulation}")

        variables_df = pd.DataFrame(ft.variables, index=['Description']).T
        variables_df.index.name = 'Variable'
        self.container.table(variables_df)
        self.container.text(f"Factor resources: {ft.factor_resources}")


class ModelTaskWindow(StWindow):

    def consume_msg(self, msg: Message):
        mt: ModelTask = msg.content

        self.container.markdown(f"**Model Name**: {mt.name}")
        self.container.markdown(f"**Model Type**: {mt.model_type}")
        self.container.markdown(f"**Description**: {mt.description}")
        self.container.markdown(f"**Formulation**: {mt.formulation}")
        
        variables_df = pd.DataFrame(mt.variables, index=['Value']).T
        variables_df.index.name = 'Variable'
        self.container.table(variables_df)


class FactorFeedbackWindow(StWindow):

    def consume_msg(self, msg: Message):
        fb: FactorSingleFeedback = msg.content
        self.container.markdown(f"""### :blue[Factor Execution Feedback]
{fb.execution_feedback}
### :blue[Factor Code Feedback]
{fb.code_feedback}
### :blue[Factor Value Feedback]
{fb.factor_value_feedback}
### :blue[Factor Final Feedback]
{fb.final_feedback}
### :blue[Factor Final Decision]
This implementation is {'SUCCESS' if fb.final_decision else 'FAIL'}.
""")


class ModelFeedbackWindow(StWindow):

    def consume_msg(self, msg: Message):
        mb: ModelCoderFeedback = msg.content
        self.container.markdown(f"""### :blue[Model Execution Feedback]
{mb.execution_feedback}
### :blue[Model Shape Feedback]
{mb.shape_feedback}
### :blue[Model Value Feedback]
{mb.value_feedback}
### :blue[Model Code Feedback]
{mb.code_feedback}
### :blue[Model Final Feedback]
{mb.final_feedback}
### :blue[Model Final Decision]
This implementation is {'SUCCESS' if mb.final_decision else 'FAIL'}.
""")


class WorkspaceWindow(StWindow):

    def consume_msg(self, msg: Message):
        ws: FactorFBWorkspace | ModelFBWorkspace = msg.content

        # no workspace
        if ws is None: return

        # task info
        task_msg = deepcopy(msg)
        task_msg.content = ws.target_task
        if isinstance(ws, FactorFBWorkspace):
            self.container.subheader('Factor Info')
            FactorTaskWindow(self.container.container()).consume_msg(task_msg)
        else:
            self.container.subheader('Model Info')
            ModelTaskWindow(self.container.container()).consume_msg(task_msg)

        # task codes
        self.container.subheader('Codes')
        for k,v in ws.code_dict.items():
            self.container.markdown(f"`{k}`")
            self.container.code(v, language="python")

        # executed_factor_value_dataframe
        if isinstance(ws, FactorFBWorkspace):
            self.container.subheader('Executed Factor Value Dataframe')
            self.container.dataframe(ws.executed_factor_value_dataframe)


class QlibFactorExpWindow(StWindow):

    def consume_msg(self, msg: Message):
        exp: QlibFactorExperiment = msg.content

        # factor tasks
        ftm_msg = deepcopy(msg)
        ftm_msg.content = [ws for ws in exp.sub_workspace_list if ws]
        ObjectsTabsWindow(self.container.expander('Factor Tasks'),
                          inner_class=WorkspaceWindow,
                          mapper=lambda x: x.target_task.factor_name,
                          ).consume_msg(ftm_msg)

        # result
        self.container.subheader('Results', divider=True)
        results = pd.DataFrame({f'base_exp_{id}':e.result for id, e in enumerate(exp.based_experiments)})
        results['now'] = exp.result

        self.container.expander('results table').table(results)

        try:
            bar_chart = px.bar(results, orientation='h', barmode='group')
            self.container.expander('results chart').plotly_chart(bar_chart)
        except:
            self.container.text('Results are incomplete.')


class QlibModelExpWindow(StWindow):

    def consume_msg(self, msg: Message):
        exp: QlibModelExperiment = msg.content

        # model tasks
        _msg = deepcopy(msg)
        _msg.content = [ws for ws in exp.sub_workspace_list if ws]
        ObjectsTabsWindow(self.container.expander('Model Tasks'),
                          inner_class=WorkspaceWindow,
                          mapper=lambda x: x.target_task.name,
                          ).consume_msg(_msg)

        # result
        self.container.subheader('Results', divider=True)
        results = pd.DataFrame({f'base_exp_{id}':e.result for id, e in enumerate(exp.based_experiments)})
        results['now'] = exp.result

        self.container.expander('results table').table(results)


class QlibTraceWindow(StWindow):

    def __init__(self, container: 'DeltaGenerator' = st.container(), show_llm: bool = False, show_common_logs: bool = True):
        super().__init__(container)
        self.show_llm = show_llm
        self.show_common_logs = show_common_logs
        self.pid_trace = ''
        self.current_tag = ''

        self.current_win = StWindow(self.container)
        self.evolving_tasks: list[str] = []

    def consume_msg(self, msg: Message):

        # divide tag levels
        if len(msg.tag) > len(self.current_tag):
            # write a header about current task, if it is llm message, not write.
            if not msg.tag.endswith('llm_messages'):
                self.container.header(msg.tag.replace('.', ' ➡ '), divider=True)
        
        self.current_tag = msg.tag

        # set log writer (window) according to msg
        if msg.tag.endswith('llm_messages'):
            # llm messages logs
            if not self.show_llm:
                return
            if not isinstance(self.current_win, LLMWindow):
                self.current_win = LLMWindow(self.container)
        elif isinstance(msg.content, Hypothesis):
            # hypothesis
            self.current_win = HypothesisWindow(self.container)
        elif isinstance(msg.content, HypothesisFeedback):
            # hypothesis feedback
            self.current_win = HypothesisFeedbackWindow(self.container)
        elif isinstance(msg.content, QlibFactorExperiment):
            self.current_win = QlibFactorExpWindow(self.container)
        elif isinstance(msg.content, QlibModelExperiment):
            self.current_win = QlibModelExpWindow(self.container)
        elif isinstance(msg.content, list):
            msg.content = [m for m in msg.content if m]
            if len(msg.content) == 0:
                return
            if isinstance(msg.content[0], FactorTask):
                self.current_win = ObjectsTabsWindow(self.container.expander('Factor Tasks'), FactorTaskWindow, lambda x: x.factor_name)
            elif isinstance(msg.content[0], ModelTask):
                self.current_win = ObjectsTabsWindow(self.container.expander('Model Tasks'), ModelTaskWindow, lambda x: x.name)
            
            elif isinstance(msg.content[0], FactorFBWorkspace):
                self.current_win = ObjectsTabsWindow(self.container.expander('Factor Workspaces'),
                                                        inner_class=WorkspaceWindow,
                                                        mapper=lambda x: x.target_task.factor_name)
                self.evolving_tasks = [m.target_task.factor_name for m in msg.content]
            elif isinstance(msg.content[0], ModelFBWorkspace):
                self.current_win = ObjectsTabsWindow(self.container.expander('Model Workspaces'),
                                                     inner_class=WorkspaceWindow,
                                                     mapper=lambda x: x.target_task.name)
                self.evolving_tasks = [m.target_task.name for m in msg.content]

            elif isinstance(msg.content[0], FactorSingleFeedback):
                self.current_win = ObjectsTabsWindow(self.container.expander('Factor Feedbacks'),
                                                        inner_class=FactorFeedbackWindow,
                                                        tab_names=self.evolving_tasks)
            elif isinstance(msg.content[0], ModelCoderFeedback):
                self.current_win = ObjectsTabsWindow(self.container.expander('Model Feedbacks'),
                                                     inner_class=ModelFeedbackWindow,
                                                     tab_names=self.evolving_tasks)
        else:
            # common logs
            if not self.show_common_logs:
                return
            self.current_win = StWindow(self.container)

        self.current_win.consume_msg(msg)


def mock_msg(obj) -> Message:
    return Message(tag='mock', level='INFO', timestamp=datetime.now(), pid_trace='000', caller='mock',content=obj)


from rdagent.core.proposal import Trace
class TraceObjWindow(StWindow):

    def __init__(self, container: 'DeltaGenerator' = st.container()):
        self.container = container

    def consume_msg(self, msg: Message):
        trace:Trace = msg.content

        for id, (h, e, hf) in enumerate(trace.hist):
            self.container.header(f'Trace History {id}', divider=True)
            HypothesisWindow(self.container).consume_msg(mock_msg(h))
            if isinstance(e, QlibFactorExperiment):
                QlibFactorExpWindow(self.container).consume_msg(mock_msg(e))
            else:
                QlibModelExpWindow(self.container).consume_msg(mock_msg(e))
            HypothesisFeedbackWindow(self.container).consume_msg(mock_msg(hf))

