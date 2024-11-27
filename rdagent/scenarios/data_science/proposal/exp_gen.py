from argparse import ONE_OR_MORE
from typing import Literal
from rdagent.components.proposal import LLMHypothesis2Experiment, LLMHypothesisGen
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import ExpGen, Trace
from rdagent.core.scenario import Scenario
from rdagent.utils.agent.tpl import T
from rdagent.oai.llm_utils import APIBackend
from rdagent.components.coder.data_science.raw_data_loader.raw_data_loader import (
    DataLoaderExperiment,
    DataLoaderFBWorkspace,
    DataLoaderTask,
) 

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
                    # TODO return a description of the data loading task
                    system = T(".prompts:DataLoaderSpec.system").r()
                    user  = T(".prompts:DataLoaderSpec.user").r()
                    data_load_exp = APIBackend().build_messages_and_create_chat_completion(
                        user_prompt=user,
                        system_prompt=system,
                        json_mode=True,
                    )
                    dlt = DataLoaderTask(name="DataLoaderTask", description=data_load_exp)
                    exp = DataLoaderExperiment(
                        sub_tasks=[dlt],
                    )
                    return exp
                else:
                    ... # two components
        return super().gen(trace)

