from rdagent.log.base import View, Storage
from pathlib import Path

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
        pass