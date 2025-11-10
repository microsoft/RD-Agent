from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment

# TODO:  We only list the dummy coder and runner here.
# If we want to implement the a comprehensive agentic system R&D Agent, we need to implement it with CoSTEER.


class AgenticSysCoder(Developer[Experiment]):

    def develop(self, exp: Experiment) -> Experiment:
        # TODO: implement the coder
        return exp


class AgenticSysRunner(Developer[Experiment]):

    def develop(self, exp: Experiment) -> Experiment:
        # TODO: implement the runner
        return exp
