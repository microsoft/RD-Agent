from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Evaluator, Feedback
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Task, Workspace
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    from rdagent.core.scenario import Scenario

# TODO:
# 1. It seems logically sound, but we currently lack a scenario to apply it.
# 2. If it proves to be useful, relocate it to a more general location.
#
# class FBWorkspaceExeFeedback(Feedback):
#     """
#     It pairs with FBWorkspace in the abstract level.
#     """
#     # ws: FBWorkspace   # potential
#     stdout: str


@dataclass
class CoSTEERSingleFeedback(Feedback):
    # TODO: (xiao)
    # it should be more general class for FBWorkspaceExeFeedback
    # A better name of it may be NormalFeedback
    # TODO: It should be a general feeddback for CoSTEERR
    """
    The feedback for the data loader evaluation.
    It is design align the phases of the implemented code
    - Execution -> Return Value -> Code -> Final Decision
    """
    execution: str
    # execution_feedback
    return_checking: str | None  # including every check in the testing (constraints about the generated value)
    # value_feedback, shape_feedback, value_generated_flag
    code: str
    final_decision: bool | None = None

    @staticmethod
    def val_and_update_init_dict(data: dict) -> dict:
        # TODO: (bowen) use a more general method to validate and update the data dictionary before init, like pydantic
        """
        Validates and converts the 'final_decision' field in the given data dictionary.

        Args:
            data (dict): The data dictionary containing the 'final_decision' field.

        Returns:
            dict: The updated data dictionary with 'final_decision' as a boolean.

        Raises:
            ValueError: If 'final_decision' is not present or not a boolean.
        """
        if "final_decision" not in data:
            raise ValueError("'final_decision' is required")

        if isinstance(data["final_decision"], str):
            if data["final_decision"] == "false" or data["final_decision"] == "False":
                data["final_decision"] = False
            elif data["final_decision"] == "true" or data["final_decision"] == "True":
                data["final_decision"] = True

        if not isinstance(data["final_decision"], bool):
            raise ValueError(f"'final_decision' must be a boolean, not {type(data['final_decision'])}")

        for attr in "execution", "return_checking", "code":
            if data[attr] is not None and not isinstance(data[attr], str):
                raise ValueError(f"'{attr}' must be a string, not {type(data[attr])}")
        return data

    @classmethod
    def merge(cls, feedback_li: list["CoSTEERSingleFeedback"]) -> "CoSTEERSingleFeedback":
        # NOTE:
        # Here we don't know the detailed design of each feedback, we just know they are CoSTEERSingleFeedback
        # So we merge them only based on CoSTEERSingleFeedback's attributes
        # **So some information may be lost when we have different types of feedbacks**
        # If you have more sophisticated sub class of CoSTEERSingleFeedback, you should override this method
        # to avoid the loss of information.

        fb = deepcopy(feedback_li[0])

        # for all the evaluators, aggregate the final_decision from `task_id`
        fb.final_decision = all(fb.final_decision for fb in feedback_li)
        for attr in "execution", "return_checking", "code":
            setattr(
                fb,
                attr,
                "\n\n".join([getattr(_fb, attr) for _fb in feedback_li if getattr(_fb, attr) is not None]),
            )
        return fb

    def __str__(self) -> str:
        return f"""------------------Execution------------------
{self.execution}
------------------Return Checking------------------
{self.return_checking if self.return_checking is not None else 'No return checking'}
------------------Code------------------
{self.code}
------------------Final Decision------------------
This implementation is {'SUCCESS' if self.final_decision else 'FAIL'}.
"""

    def __bool__(self):
        return self.final_decision


class CoSTEERSingleFeedbackDeprecated(CoSTEERSingleFeedback):
    """This class is a base class for all code generator feedback to single implementation"""

    def __init__(
        self,
        execution_feedback: str = None,
        shape_feedback: str = None,
        code_feedback: str = None,
        value_feedback: str = None,
        final_decision: bool = None,
        final_feedback: str = None,
        value_generated_flag: bool = None,
        final_decision_based_on_gt: bool = None,
    ) -> None:
        self.execution_feedback = execution_feedback
        self.code_feedback = code_feedback
        self.value_feedback = value_feedback
        self.final_decision = final_decision
        self.final_feedback = final_feedback
        self.value_generated_flag = value_generated_flag
        self.final_decision_based_on_gt = final_decision_based_on_gt

        # TODO:
        # Not general enough. So we should not put them in the general costeer feedback
        # Instead, we should create subclass for it.
        self.shape_feedback = shape_feedback  # Not general enough. So

    @property
    def execution(self):
        return self.execution_feedback

    @execution.setter
    def execution(self, value):
        self.execution_feedback = value

    @property
    def return_checking(self):
        if self.value_generated_flag:
            return f"value feedback: {self.value_feedback}\n\nshape feedback: {self.shape_feedback}"
        return None

    @return_checking.setter
    def return_checking(self, value):
        # Since return_checking is derived from value_feedback and shape_feedback,
        # we don't need to do anything here
        self.value_feedback = value
        self.shape_feedback = value

    @property
    def code(self):
        return self.code_feedback

    @code.setter
    def code(self, value):
        self.code_feedback = value

    def __str__(self) -> str:
        return f"""------------------Execution Feedback------------------
{self.execution_feedback if self.execution_feedback is not None else 'No execution feedback'}
------------------Shape Feedback------------------
{self.shape_feedback if self.shape_feedback is not None else 'No shape feedback'}
------------------Code Feedback------------------
{self.code_feedback if self.code_feedback is not None else 'No code feedback'}
------------------Value Feedback------------------
{self.value_feedback if self.value_feedback is not None else 'No value feedback'}
------------------Final Feedback------------------
{self.final_feedback if self.final_feedback is not None else 'No final feedback'}
------------------Final Decision------------------
This implementation is {'SUCCESS' if self.final_decision else 'FAIL'}.
"""


class CoSTEERMultiFeedback(Feedback):
    """Feedback contains a list, each element is the corresponding feedback for each factor implementation."""

    def __init__(self, feedback_list: List[CoSTEERSingleFeedback]) -> None:
        self.feedback_list = feedback_list

    def __getitem__(self, index: int) -> CoSTEERSingleFeedback:
        return self.feedback_list[index]

    def __len__(self) -> int:
        return len(self.feedback_list)

    def append(self, feedback: CoSTEERSingleFeedback) -> None:
        self.feedback_list.append(feedback)

    def __iter__(self):
        return iter(self.feedback_list)

    def is_acceptable(self) -> bool:
        return all(feedback.is_acceptable() for feedback in self.feedback_list)

    def finished(self) -> bool:
        """
        In some implementations, tasks may fail multiple times, leading agents to skip the implementation.
        This results in None feedback. However, we want to accept the correct parts and ignore None feedback.
        """
        return all(feedback.final_decision for feedback in self.feedback_list if feedback is not None)

    def __bool__(self) -> bool:
        return all(feedback.final_decision for feedback in self.feedback_list)


class CoSTEEREvaluator(Evaluator):
    def __init__(
        self,
        scen: "Scenario",
    ) -> None:
        self.scen = scen

    # TODO:
    # I think we should have unified interface for all evaluates, for examples.
    # So we should adjust the interface of other factors
    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        raise NotImplementedError("Please implement the `evaluator` method")


class CoSTEERMultiEvaluator(CoSTEEREvaluator):
    """This is for evaluation of experiment. Due to we have multiple tasks, so we will return a list of evaluation feebacks"""

    def __init__(self, single_evaluator: CoSTEEREvaluator | list[CoSTEEREvaluator], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.single_evaluator = single_evaluator

    def evaluate(
        self,
        evo: EvolvingItem,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERMultiFeedback:
        eval_l = self.single_evaluator if isinstance(self.single_evaluator, list) else [self.single_evaluator]

        # 1) Evaluate each sub_task
        task_li_feedback_li = []
        # task_li_feedback_li: List[List[CoSTEERSingleFeedback]]
        # Example:
        # If there are 2 evaluators and 3 sub_tasks in evo, and each evaluator's evaluate returns a list of 3 CoSTEERSingleFeedbacks,
        # Then task_li_feedback_li will be:
        # [
        #   [feedback_1_1, feedback_1_2, feedback_1_3],  # results from the 1st evaluator for all sub_tasks
        #   [feedback_2_1, feedback_2_2, feedback_2_3],  # results from the 2nd evaluator for all sub_tasks
        # ]
        # Where feedback_i_j is the feedback from the i-th evaluator for the j-th sub_task.
        for ev in eval_l:
            multi_implementation_feedback = multiprocessing_wrapper(
                [
                    (
                        ev.evaluate,
                        (
                            evo.sub_tasks[index],
                            evo.sub_workspace_list[index],
                            evo.sub_gt_implementations[index] if evo.sub_gt_implementations is not None else None,
                            queried_knowledge,
                        ),
                    )
                    for index in range(len(evo.sub_tasks))
                ],
                n=RD_AGENT_SETTINGS.multi_proc_n,
            )
            task_li_feedback_li.append(multi_implementation_feedback)

        # 2) merge the feedbacks along the sub_tasks to aggregate the multiple evaluation feedbacks
        merged_task_feedback = []
        # task_li_feedback_li[0] is a list of feedbacks of different tasks for the 1st evaluator
        for task_id, fb in enumerate(task_li_feedback_li[0]):
            fb = fb.merge([fb_li[task_id] for fb_li in task_li_feedback_li])
            merged_task_feedback.append(fb)
        # merged_task_feedback: List[CoSTEERSingleFeedback]
        # Example:
        # [
        #   CoSTEERSingleFeedback(final_decision=True, execution="...", return_checking="...", code="..."),
        #   CoSTEERSingleFeedback(final_decision=False, execution="...", return_checking="...", code="..."),
        #   ...
        # ]
        # Each element corresponds to the merged feedback for one sub-task across all evaluators.
        # merged_task_feedback[i] is the merged feedback for the i-th sub_task

        final_decision = [
            None if single_feedback is None else single_feedback.final_decision
            for single_feedback in merged_task_feedback
        ]
        logger.info(f"Final decisions: {final_decision} True count: {final_decision.count(True)}")

        # TODO: this is to be compatible with factor_implementation;
        for index in range(len(evo.sub_tasks)):
            if final_decision[index]:
                evo.sub_tasks[index].factor_implementation = True

        return CoSTEERMultiFeedback(merged_task_feedback)
