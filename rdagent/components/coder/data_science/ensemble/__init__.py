"""
File structure
- ___init__.py: the entrance/agent of coder
- evaluator.py
- conf.py
- exp.py: everything under the experiment, e.g.
    - Task
    - Experiment
    - Workspace
- test.py
    - Each coder could be tested.
"""

import json

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.data_science.ensemble.eval import (
    EnsembleCoSTEEREvaluator,
)
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class EnsembleMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: EnsembleTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
    ) -> dict[str, str]:
        # return a workspace with "ensemble.py" inside
        competition_info = self.scen.competition_descriptions
        ensemble_spec = target_task.spec
        # Generate code
        system_prompt = T(".prompts:ensemble_coder.system").r(competition_info=competition_info)
        user_prompt = T(".prompts:ensemble_coder.user").r(ensemble_spec=ensemble_spec)

        ensemble_code = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
        )["code"]

        return {
            "ensemble.py": ensemble_code,
        }


class EnsembleCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        eva = CoSTEERMultiEvaluator(
            EnsembleCoSTEEREvaluator(scen=scen), scen=scen
        )
        es = EnsembleMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)

        super().__init__(
            *args,
            settings=CoSTEER_SETTINGS,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            **kwargs
        )


