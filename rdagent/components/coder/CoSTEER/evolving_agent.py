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
        # FIXME: the list does not align with the annotation; It should be MultipleFeedback instead of a list of feedbacks
        assert isinstance(feedback, list)
        assert len(evo.sub_workspace_list) == len(feedback)

        for index in range(len(evo.sub_workspace_list)):
            if evo.sub_workspace_list[index] is not None and feedback[index] is not None and not feedback[index]:
                evo.sub_workspace_list[index].clear()

        failed_feedbacks = [
            f"- feedback{index + 1:02d}:\n  - execution: {f.execution}\n  - return_checking: {f.return_checking}\n  - code: {f.code}"
            for index, f in enumerate(feedback)
            if f is not None and not f.final_decision
        ]

        if len(failed_feedbacks) == len(feedback):
            feedback_summary = "\n".join(failed_feedbacks)
            raise CoderError(f"All tasks are failed:\n{feedback_summary}")

        return evo
