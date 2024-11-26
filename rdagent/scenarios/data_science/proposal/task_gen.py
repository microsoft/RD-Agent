from typing import Literal
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import ExpGen, Trace

COMPONENT = Literal["DataLoadSpec", "FeatureEng", "Model", "Workflow", "Ensemble"]
MAX_NUM = COMPONENT.__args__

class DSExpGen(ExpGen):
    """Data Science Task Generator."""
    def __init__(self) -> None:
        self.complete_component: set[COMPONENT] = set()  # Initialize as an empty set

    def _is_complete(self):
        """is all components complete"""
        # TODO: place it into ExpGen
        return self.complete_component  == set(COMPONENT.__args__)

    def gen(self, trace: Trace) -> Experiment:
        
        return super().gen(trace)

