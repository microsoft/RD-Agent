from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.core.scenario import Scenario
from rdagent.components.coder.CoSTEER.evolving_strategy import MultiProcessEvolvingStrategy
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge

# from rdagent.utils.agent.tpl import T
# T(".prompts:model_generator.user").r()

class ModelMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: ModelTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
    ) -> str:
        return """
        import pandas as pd
        def Model():
            pass
        """
    
    def assign_code_list_to_evo(self, code_list: list, evo) -> None:
        """
        Assign the code list to the evolving item.

        The code list is aligned with the evolving item's sub-tasks.
        If a task is not implemented, put a None in the list.
        """
        raise NotImplementedError


class ModelCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        eva = CoSTEERMultiEvaluator(
            # ModelCoSTEEREvaluator(scen=scen), scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        es = ModelMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)

        super().__init__(*args, settings=CoSTEER_SETTINGS, eva=eva, es=es, evolving_version=1, scen=scen, **kwargs)
