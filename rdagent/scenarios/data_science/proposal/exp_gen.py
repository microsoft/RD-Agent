from argparse import ONE_OR_MORE
from typing import Literal
from rdagent.components.proposal import LLMHypothesis2Experiment, LLMHypothesisGen
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import ExpGen, Trace
from rdagent.core.scenario import Scenario
from rdagent.utils.agent.tpl import T

COMPONENT = Literal["DataLoadSpec", "FeatureEng", "Model", "Workflow", "Ensemble"]
ORDER = COMPONENT.__args__


class DSExpGen(ExpGen):
    """Data Science Task Generator."""
    def __init__(self, scen: Scenario) -> None:
        self.complete_component: set[COMPONENT] = set()  # Initialize as an empty set
        super().__init__(scen)

    def is_complete(self):
        """is all components complete"""
        # TODO: place it into ExpGen
        return self.complete_component  == set(COMPONENT.__args__)

    def gen(self, trace: Trace) -> Experiment:
        if self.is_complete():
            # proposal + design
            pass
            # TODO: We can create subclasses for them if we need two components
            LLMHypothesisGen
            LLMHypothesis2Experiment
        else:
            #         
            for o in ORDER:
                if o in self.complete_component:
                    continue
                elif o == "DataLoadSpec":
                    system = T(".prompts:DataLoadSpec.system").r()
                    user  = T(".prompts:DataLoadSpec.user").r()
                else:
                    ... # two components
        return super().gen(trace)

