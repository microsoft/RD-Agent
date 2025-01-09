from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedbackDeprecated
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.core.evolving_agent import RAGEvoAgent
from rdagent.core.evolving_framework import EvolvableSubjects
from rdagent.core.exception import CoderError


class FilterFailedRAGEvoAgent(RAGEvoAgent):
    def filter_evolvable_subjects_by_feedback(
        self, evo: EvolvableSubjects, feedback: CoSTEERSingleFeedbackDeprecated
    ) -> EvolvableSubjects:
        assert isinstance(evo, EvolvingItem)
        assert isinstance(feedback, list)
        assert len(evo.sub_workspace_list) == len(feedback)

        for index in range(len(evo.sub_workspace_list)):
            if evo.sub_workspace_list[index] is not None and feedback[index] is not None and not feedback[index]:
                evo.sub_workspace_list[index].clear()

        failed_feedbacks = [
            f"- trial{index + 1}:\n  - feedback:\n    - execution: {f.execution}\n    - return_checking: {f.return_checking}\n    - code: {f.code}"
            for index, f in enumerate(feedback) if f and not f.final_decision
        ]

        if failed_feedbacks:
            feedback_summary = "\n".join(failed_feedbacks)
            raise CoderError(f"All tasks are failed:\n{feedback_summary}")

        return evo
