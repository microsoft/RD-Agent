from rdagent.components.coder.factor_coder.CoSTEER.evaluators import FactorMultiFeedback
from rdagent.components.coder.factor_coder.CoSTEER.evolvable_subjects import (
    FactorEvolvingItem,
)
from rdagent.core.evaluation import Feedback
from rdagent.core.evolving_agent import RAGEvoAgent
from rdagent.core.evolving_framework import EvolvableSubjects


class FactorRAGEvoAgent(RAGEvoAgent):
    def filter_evolvable_subjects_by_feedback(self, evo: EvolvableSubjects, feedback: Feedback) -> EvolvableSubjects:
        assert isinstance(evo, FactorEvolvingItem)
        assert isinstance(feedback, list)
        assert len(evo.sub_workspace_list) == len(feedback)

        for index in range(len(evo.sub_workspace_list)):
            if not feedback[index].final_decision:
                evo.sub_workspace_list[index].clear()
        return evo
