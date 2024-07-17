from rdagent.components.coder.model_coder.CoSTEER.evaluators import ModelCoderFeedback
from rdagent.components.coder.model_coder.CoSTEER.evolvable_subjects import (
    ModelEvolvingItem,
)
from rdagent.core.evaluation import Feedback
from rdagent.core.evolving_agent import RAGEvoAgent
from rdagent.core.evolving_framework import EvolvableSubjects


class ModelRAGEvoAgent(RAGEvoAgent):
    def filter_evolvable_subjects_by_feedback(self, evo: EvolvableSubjects, feedback: Feedback) -> EvolvableSubjects:
        assert isinstance(evo, ModelEvolvingItem)
        assert isinstance(feedback, list)
        assert len(evo.sub_workspace_list) == len(feedback)

        for index in range(len(evo.sub_workspace_list)):
            if not feedback[index].final_decision:
                evo.sub_workspace_list[index].clear()
        return evo
