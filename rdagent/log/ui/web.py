from pathlib import Path

from rdagent.log.base import Storage, View


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

    def __init__(self, trace_path: Path):
        pass
        # Save logs to your desired data structure
        # ...

    def display(s: Storage, watch: bool = False):
        ui = STLUI()
        for msg in s.iter_msg():  # iterate overtime
            # NOTE:  iter_msg will correctly seperate the information.
            # TODO: msg may support streaming mode.
            ui.dispatch(msg)
        pass


# TODO: Implement the following classes
class STLWindow:
    ...

    def consume_msg(self, msg):
        ...  # update it's view


class STLUI:
    wd_l: list[STLWindow]

    def __init__(self):
        self.build_ui()

    def build_ui(self):
        # control the dispaly of windows
        ...

    def dispatch(self, msg):
        # map the message to a specific window
        ...
