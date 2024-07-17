from rdagent.components.coder.model_coder.conf import MODEL_IMPL_SETTINGS
from rdagent.components.coder.model_coder.CoSTEER.evaluators import ModelCoderFeedback
from rdagent.components.coder.model_coder.model import ModelTask
from rdagent.core.evolving_framework import (
    EvolvableSubjects,
    EvoStep,
    Knowledge,
    KnowledgeBase,
    QueriedKnowledge,
    RAGStrategy,
)
from rdagent.core.experiment import Workspace
from rdagent.oai.llm_utils import calculate_embedding_distance_between_str_list


class ModelKnowledge(Knowledge):
    def __init__(
        self,
        target_task: ModelTask,
        implementation: Workspace,
        feedback: ModelCoderFeedback,
    ) -> None:
        """
        Initialize a ModelKnowledge object. The ModelKnowledge object is used to store a model implementation without the ground truth code and value.

        Args:
            model (Model): The model object associated with the KnowledgeManagement.

        Returns:
            None
        """
        self.target_task = target_task
        self.implementation = implementation.copy()
        self.feedback = feedback

    def get_implementation_and_feedback_str(self) -> str:
        return f"""------------------Model implementation code:------------------
{self.implementation.code}
------------------Model implementation feedback:------------------
{self.feedback!s}
"""


class ModelQueriedKnowledge(QueriedKnowledge):
    def __init__(self, success_task_to_knowledge_dict: dict = {}, failed_task_info_set: set = set()) -> None:
        self.success_task_to_knowledge_dict = success_task_to_knowledge_dict
        self.failed_task_info_set = failed_task_info_set
        self.working_task_to_former_failed_knowledge_dict = dict()
        self.working_task_to_similar_successful_knowledge_dict = dict()


class ModelKnowledgeBase(KnowledgeBase):
    def __init__(self) -> None:
        self.implementation_trace: dict[str, ModelKnowledge] = dict()
        self.success_task_info_set: set[str] = set()

        self.task_to_embedding = dict()

    def query(self) -> QueriedKnowledge | None:
        """
        Query the knowledge base to get the queried knowledge. So far is handled in RAG strategy.
        """
        raise NotImplementedError


class ModelRAGStrategy(RAGStrategy):
    def __init__(self, knowledgebase: ModelKnowledgeBase) -> None:
        super().__init__(knowledgebase)
        self.current_generated_trace_count = 0

    def generate_knowledge(
        self,
        evolving_trace: list[EvoStep],
        *,
        return_knowledge: bool = False,
    ) -> Knowledge | None:
        if len(evolving_trace) == self.current_generated_trace_count:
            return
        else:
            for trace_index in range(
                self.current_generated_trace_count,
                len(evolving_trace),
            ):
                evo_step = evolving_trace[trace_index]
                implementations = evo_step.evolvable_subjects
                feedback = evo_step.feedback
                for task_index in range(len(implementations.sub_tasks)):
                    target_task = implementations.sub_tasks[task_index]
                    target_task_information = target_task.get_task_information()
                    implementation = implementations.sub_workspace_list[task_index]
                    single_feedback = feedback[task_index]
                    if single_feedback is None:
                        continue
                    single_knowledge = ModelKnowledge(
                        target_task=target_task,
                        implementation=implementation,
                        feedback=single_feedback,
                    )
                    if target_task_information not in self.knowledgebase.success_task_info_set:
                        self.knowledgebase.implementation_trace.setdefault(
                            target_task_information,
                            [],
                        ).append(single_knowledge)

                        if single_feedback.final_decision == True:
                            self.knowledgebase.success_task_info_set.add(
                                target_task_information,
                            )
            self.current_generated_trace_count = len(evolving_trace)

    def query(
        self,
        evo: EvolvableSubjects,
        evolving_trace: list[EvoStep],
    ) -> QueriedKnowledge | None:
        query_former_trace_limit = MODEL_IMPL_SETTINGS.query_former_trace_limit
        query_similar_success_limit = MODEL_IMPL_SETTINGS.query_similar_success_limit
        fail_task_trial_limit = MODEL_IMPL_SETTINGS.fail_task_trial_limit

        queried_knowledge = ModelQueriedKnowledge()
        for target_model_task in evo.sub_tasks:
            target_model_task_information = target_model_task.get_task_information()
            if target_model_task_information in self.knowledgebase.success_task_info_set:
                queried_knowledge.success_task_to_knowledge_dict[
                    target_model_task_information
                ] = self.knowledgebase.implementation_trace[target_model_task_information][-1]
            elif (
                len(
                    self.knowledgebase.implementation_trace.setdefault(
                        target_model_task_information,
                        [],
                    ),
                )
                >= fail_task_trial_limit
            ):
                queried_knowledge.failed_task_info_set.add(target_model_task_information)
            else:
                queried_knowledge.working_task_to_former_failed_knowledge_dict[
                    target_model_task_information
                ] = self.knowledgebase.implementation_trace.setdefault(
                    target_model_task_information,
                    [],
                )[
                    -query_former_trace_limit:
                ]

                knowledge_base_success_task_list = list(
                    self.knowledgebase.success_task_info_set,
                )
                similarity = calculate_embedding_distance_between_str_list(
                    [target_model_task_information],
                    knowledge_base_success_task_list,
                )[0]
                similar_indexes = sorted(
                    range(len(similarity)),
                    key=lambda i: similarity[i],
                    reverse=True,
                )[:query_similar_success_limit]
                similar_successful_knowledge = [
                    self.knowledgebase.implementation_trace.setdefault(
                        knowledge_base_success_task_list[index],
                        [],
                    )[-1]
                    for index in similar_indexes
                ]
                queried_knowledge.working_task_to_similar_successful_knowledge_dict[
                    target_model_task_information
                ] = similar_successful_knowledge
        return queried_knowledge
