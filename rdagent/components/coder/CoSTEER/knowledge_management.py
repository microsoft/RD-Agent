from __future__ import annotations

import copy
import json
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Union

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode,
)
from rdagent.core.evolving_agent import Feedback
from rdagent.core.evolving_framework import (
    EvolvableSubjects,
    EvolvingKnowledgeBase,
    EvoStep,
    Knowledge,
    QueriedKnowledge,
    RAGStrategy,
)
from rdagent.core.experiment import FBWorkspace
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Task
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import (
    APIBackend,
    calculate_embedding_distance_between_str_list,
)


class CoSTEERKnowledge(Knowledge):
    def __init__(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        feedback: Feedback,
    ) -> None:
        self.target_task = target_task
        self.implementation = implementation.copy()
        self.feedback = feedback

    def get_implementation_and_feedback_str(self) -> str:
        return f"""------------------implementation code:------------------
{self.implementation.code}
------------------implementation feedback:------------------
{self.feedback!s}
"""


class CoSTEERQueriedKnowledge(QueriedKnowledge):
    def __init__(self, success_task_to_knowledge_dict: dict = {}, failed_task_info_set: set = set()) -> None:
        self.success_task_to_knowledge_dict = success_task_to_knowledge_dict
        self.failed_task_info_set = failed_task_info_set


class CoSTEERKnowledgeBaseV1(EvolvingKnowledgeBase):
    def __init__(self, path: str | Path = None) -> None:
        self.implementation_trace: dict[str, CoSTEERKnowledge] = dict()
        self.success_task_info_set: set[str] = set()

        self.task_to_embedding = dict()
        super().__init__(path)

    def query(self) -> CoSTEERQueriedKnowledge | None:
        """
        Query the knowledge base to get the queried knowledge. So far is handled in RAG strategy.
        """
        raise NotImplementedError


class CoSTEERQueriedKnowledgeV1(CoSTEERQueriedKnowledge):
    def __init__(
        self,
        *args,
        task_to_former_failed_traces: dict = {},
        task_to_similar_task_successful_knowledge: dict = {},
        **kwargs,
    ) -> None:
        self.task_to_former_failed_traces = task_to_former_failed_traces
        self.task_to_similar_task_successful_knowledge = task_to_similar_task_successful_knowledge
        super().__init__(*args, **kwargs)


class CoSTEERRAGStrategyV1(RAGStrategy):
    def __init__(self, knowledgebase: CoSTEERKnowledgeBaseV1, settings: CoSTEERSettings) -> None:
        super().__init__(knowledgebase)
        self.current_generated_trace_count = 0
        self.settings = settings

    def generate_knowledge(
        self,
        evolving_trace: list[EvoStep],
        *,
        return_knowledge: bool = False,
    ) -> Knowledge | None:
        raise NotImplementedError(
            "This method should be considered as an un-implemented method because we encourage everyone to use v2."
        )
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
                    single_knowledge = CoSTEERKnowledge(
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
    ) -> CoSTEERQueriedKnowledge | None:
        raise NotImplementedError(
            "This method should be considered as an un-implemented method because we encourage everyone to use v2."
        )
        v1_query_former_trace_limit = self.settings.v1_query_former_trace_limit
        v1_query_similar_success_limit = self.settings.v1_query_similar_success_limit
        fail_task_trial_limit = self.settings.fail_task_trial_limit

        queried_knowledge = CoSTEERQueriedKnowledgeV1()
        for target_task in evo.sub_tasks:
            target_task_information = target_task.get_task_information()
            if target_task_information in self.knowledgebase.success_task_info_set:
                queried_knowledge.success_task_to_knowledge_dict[target_task_information] = (
                    self.knowledgebase.implementation_trace[target_task_information][-1]
                )
            elif (
                len(
                    self.knowledgebase.implementation_trace.setdefault(
                        target_task_information,
                        [],
                    ),
                )
                >= fail_task_trial_limit
            ):
                queried_knowledge.failed_task_info_set.add(target_task_information)
            else:
                queried_knowledge.task_to_former_failed_traces[target_task_information] = (
                    self.knowledgebase.implementation_trace.setdefault(
                        target_task_information,
                        [],
                    )[-v1_query_former_trace_limit:]
                )

                knowledge_base_success_task_list = list(
                    self.knowledgebase.success_task_info_set,
                )
                similarity = calculate_embedding_distance_between_str_list(
                    [target_task_information],
                    knowledge_base_success_task_list,
                )[0]
                similar_indexes = sorted(
                    range(len(similarity)),
                    key=lambda i: similarity[i],
                    reverse=True,
                )[:v1_query_similar_success_limit]
                similar_successful_knowledge = [
                    self.knowledgebase.implementation_trace.setdefault(
                        knowledge_base_success_task_list[index],
                        [],
                    )[-1]
                    for index in similar_indexes
                ]
                queried_knowledge.task_to_similar_task_successful_knowledge[target_task_information] = (
                    similar_successful_knowledge
                )
        return queried_knowledge


class CoSTEERQueriedKnowledgeV2(CoSTEERQueriedKnowledgeV1):
    # Aggregation of knowledge
    def __init__(
        self,
        task_to_former_failed_traces: dict = {},
        task_to_similar_task_successful_knowledge: dict = {},
        task_to_similar_error_successful_knowledge: dict = {},
        **kwargs,
    ) -> None:
        self.task_to_similar_error_successful_knowledge = task_to_similar_error_successful_knowledge
        super().__init__(
            task_to_former_failed_traces=task_to_former_failed_traces,
            task_to_similar_task_successful_knowledge=task_to_similar_task_successful_knowledge,
            **kwargs,
        )


class CoSTEERRAGStrategyV2(RAGStrategy):
    prompt = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

    def __init__(self, knowledgebase: CoSTEERKnowledgeBaseV2, settings: CoSTEERSettings) -> None:
        super().__init__(knowledgebase)
        self.current_generated_trace_count = 0
        self.settings = settings

    def generate_knowledge(
        self,
        evolving_trace: list[EvoStep],
        *,
        return_knowledge: bool = False,
    ) -> Knowledge | None:
        if len(evolving_trace) == self.current_generated_trace_count:
            return None

        else:
            for trace_index in range(self.current_generated_trace_count, len(evolving_trace)):
                evo_step = evolving_trace[trace_index]
                implementations = evo_step.evolvable_subjects
                feedback = evo_step.feedback
                for task_index in range(len(implementations.sub_tasks)):
                    target_task = implementations.sub_tasks[task_index]
                    target_task_information = target_task.get_task_information()
                    implementation = implementations.sub_workspace_list[task_index]
                    single_feedback: CoSTEERSingleFeedback = feedback[task_index]
                    if implementation is None or single_feedback is None:
                        continue
                    single_knowledge = CoSTEERKnowledge(
                        target_task=target_task,
                        implementation=implementation,
                        feedback=single_feedback,
                    )
                    if (
                        target_task_information not in self.knowledgebase.success_task_to_knowledge_dict
                        and implementation is not None
                    ):
                        self.knowledgebase.working_trace_knowledge.setdefault(target_task_information, []).append(
                            single_knowledge,
                        )  # save to working trace
                        if single_feedback.final_decision == True:
                            self.knowledgebase.success_task_to_knowledge_dict.setdefault(
                                target_task_information,
                                single_knowledge,
                            )
                            # Do summary for the last step and update the knowledge graph
                            self.knowledgebase.update_success_task(
                                target_task_information,
                            )
                        else:
                            # generate error node and store into knowledge base
                            error_analysis_result = []
                            if not single_feedback.value_generated_flag:
                                error_analysis_result = self.analyze_error(
                                    single_feedback.execution_feedback,
                                    feedback_type="execution",
                                )
                            else:
                                error_analysis_result = self.analyze_error(
                                    single_feedback.value_feedback,
                                    feedback_type="value",
                                )
                            self.knowledgebase.working_trace_error_analysis.setdefault(
                                target_task_information,
                                [],
                            ).append(
                                error_analysis_result,
                            )  # save to working trace error record, for graph update

            self.current_generated_trace_count = len(evolving_trace)
            return None

    def query(self, evo: EvolvableSubjects, evolving_trace: list[EvoStep]) -> CoSTEERQueriedKnowledge | None:
        conf_knowledge_sampler = self.settings.v2_knowledge_sampler
        queried_knowledge_v2 = CoSTEERQueriedKnowledgeV2(
            success_task_to_knowledge_dict=self.knowledgebase.success_task_to_knowledge_dict,
        )

        queried_knowledge_v2 = self.former_trace_query(
            evo,
            queried_knowledge_v2,
            self.settings.v2_query_former_trace_limit,
            self.settings.v2_add_fail_attempt_to_latest_successful_execution,
        )
        queried_knowledge_v2 = self.component_query(
            evo,
            queried_knowledge_v2,
            self.settings.v2_query_component_limit,
            knowledge_sampler=conf_knowledge_sampler,
        )
        queried_knowledge_v2 = self.error_query(
            evo,
            queried_knowledge_v2,
            self.settings.v2_query_error_limit,
            knowledge_sampler=conf_knowledge_sampler,
        )
        return queried_knowledge_v2

    def analyze_component(
        self,
        target_task_information,
    ) -> list[UndirectedNode]:  # Hardcode: certain component nodes
        all_component_nodes = self.knowledgebase.graph.get_all_nodes_by_label_list(["component"])
        if not len(all_component_nodes):
            return []
        all_component_content = ""
        for _, component_node in enumerate(all_component_nodes):
            all_component_content += f"{component_node.content}, \n"
        analyze_component_system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(self.prompt["analyze_component_prompt_v1_system"])
            .render(
                all_component_content=all_component_content,
            )
        )

        analyze_component_user_prompt = target_task_information
        try:
            component_no_list = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    system_prompt=analyze_component_system_prompt,
                    user_prompt=analyze_component_user_prompt,
                    json_mode=True,
                ),
            )["component_no_list"]
            return [all_component_nodes[index - 1] for index in sorted(list(set(component_no_list)))]
        except:
            logger.warning("Error when analyzing components.")
            analyze_component_user_prompt = "Your response is not a valid component index list."

        return []

    def analyze_error(
        self,
        single_feedback,
        feedback_type="execution",
    ) -> list[
        UndirectedNode | str
    ]:  # Hardcode: Raised errors, existed error nodes + not existed error nodes(here, they are strs)
        if feedback_type == "execution":
            match = re.search(
                r'File "(?P<file>.+)", line (?P<line>\d+), in (?P<function>.+)\n\s+(?P<error_line>.+)\n(?P<error_type>\w+): (?P<error_message>.+)',
                single_feedback,
            )
            if match:
                error_details = match.groupdict()
                # last_traceback = f'File "{error_details["file"]}", line {error_details["line"]}, in {error_details["function"]}\n    {error_details["error_line"]}'
                error_type = error_details["error_type"]
                error_line = error_details["error_line"]
                error_contents = [f"ErrorType: {error_type}" + "\n" + f"Error line: {error_line}"]
            else:
                error_contents = ["Undefined Error"]
        elif feedback_type == "value":  # value check error
            value_check_types = r"The source dataframe and the ground truth dataframe have different rows count.|The source dataframe and the ground truth dataframe have different index.|Some values differ by more than the tolerance of 1e-6.|No sufficient correlation found when shifting up|Something wrong happens when naming the multi indices of the dataframe."
            error_contents = re.findall(value_check_types, single_feedback)
        else:
            error_contents = ["Undefined Error"]

        all_error_nodes = self.knowledgebase.graph.get_all_nodes_by_label_list(["error"])
        if not len(all_error_nodes):
            return error_contents
        else:
            error_list = []
            for error_content in error_contents:
                for error_node in all_error_nodes:
                    if error_content == error_node.content:
                        error_list.append(error_node)
                    else:
                        error_list.append(error_content)
                    if error_list[-1] in error_list[:-1]:
                        error_list.pop()

            return error_list

    def former_trace_query(
        self,
        evo: EvolvableSubjects,
        queried_knowledge_v2: CoSTEERQueriedKnowledgeV2,
        v2_query_former_trace_limit: int = 5,
        v2_add_fail_attempt_to_latest_successful_execution: bool = False,
    ) -> Union[CoSTEERQueriedKnowledge, set]:
        """
        Query the former trace knowledge of the working trace, and find all the failed task information which tried more than fail_task_trial_limit times
        """
        fail_task_trial_limit = self.settings.fail_task_trial_limit

        for target_task in evo.sub_tasks:
            target_task_information = target_task.get_task_information()
            if (
                target_task_information not in self.knowledgebase.success_task_to_knowledge_dict
                and target_task_information in self.knowledgebase.working_trace_knowledge
                and len(self.knowledgebase.working_trace_knowledge[target_task_information]) >= fail_task_trial_limit
            ):
                queried_knowledge_v2.failed_task_info_set.add(target_task_information)

            if (
                target_task_information not in self.knowledgebase.success_task_to_knowledge_dict
                and target_task_information not in queried_knowledge_v2.failed_task_info_set
                and target_task_information in self.knowledgebase.working_trace_knowledge
            ):
                former_trace_knowledge = copy.copy(
                    self.knowledgebase.working_trace_knowledge[target_task_information],
                )
                # in former trace query we will delete the right trace in the following order:[..., value_generated_flag is True, value_generated_flag is False, ...]
                # because we think this order means a deterioration of the trial (like a wrong gradient descent)
                current_index = 1
                while current_index < len(former_trace_knowledge):
                    if (
                        not former_trace_knowledge[current_index].feedback.value_generated_flag
                        and former_trace_knowledge[current_index - 1].feedback.value_generated_flag
                    ):
                        former_trace_knowledge.pop(current_index)
                    else:
                        current_index += 1

                latest_attempt = None
                if v2_add_fail_attempt_to_latest_successful_execution:
                    # When the last successful execution is not the last one in the working trace, it means we have tried to correct it. We should tell the agent this fail trial to avoid endless loop in the future.
                    if (
                        len(former_trace_knowledge) > 0
                        and len(self.knowledgebase.working_trace_knowledge[target_task_information]) > 1
                        and self.knowledgebase.working_trace_knowledge[target_task_information].index(
                            former_trace_knowledge[-1]
                        )
                        < len(self.knowledgebase.working_trace_knowledge[target_task_information]) - 1
                    ):
                        latest_attempt = self.knowledgebase.working_trace_knowledge[target_task_information][-1]

                queried_knowledge_v2.task_to_former_failed_traces[target_task_information] = (
                    former_trace_knowledge[-v2_query_former_trace_limit:],
                    latest_attempt,
                )
            else:
                queried_knowledge_v2.task_to_former_failed_traces[target_task_information] = ([], None)

        return queried_knowledge_v2

    def component_query(
        self,
        evo: EvolvableSubjects,
        queried_knowledge_v2: CoSTEERQueriedKnowledgeV2,
        v2_query_component_limit: int = 5,
        knowledge_sampler: float = 1.0,
    ) -> CoSTEERQueriedKnowledge | None:
        for target_task in evo.sub_tasks:
            target_task_information = target_task.get_task_information()
            if (
                target_task_information in self.knowledgebase.success_task_to_knowledge_dict
                or target_task_information in queried_knowledge_v2.failed_task_info_set
            ):
                queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information] = []
            else:
                if target_task_information not in self.knowledgebase.task_to_component_nodes:
                    self.knowledgebase.task_to_component_nodes[target_task_information] = self.analyze_component(
                        target_task_information,
                    )

                component_analysis_result = self.knowledgebase.task_to_component_nodes[target_task_information]

                if len(component_analysis_result) > 1:
                    task_des_node_list = self.knowledgebase.graph_query_by_intersection(
                        component_analysis_result,
                        constraint_labels=["task_description"],
                    )
                    single_component_constraint = (v2_query_component_limit // len(component_analysis_result)) + 1
                else:
                    task_des_node_list = []
                    single_component_constraint = v2_query_component_limit
                queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information] = []
                for component_node in component_analysis_result:
                    # Reverse iterate, a trade-off with intersection search
                    count = 0
                    for task_des_node in self.knowledgebase.graph_query_by_node(
                        node=component_node,
                        step=1,
                        constraint_labels=["task_description"],
                        block=True,
                    )[::-1]:
                        if task_des_node not in task_des_node_list:
                            task_des_node_list.append(task_des_node)
                            count += 1
                        if count >= single_component_constraint:
                            break

                for node in task_des_node_list:
                    for searched_node in self.knowledgebase.graph_query_by_node(
                        node=node,
                        step=50,
                        constraint_labels=[
                            "task_success_implement",
                        ],
                        block=True,
                    ):
                        if searched_node.label == "task_success_implement":
                            target_knowledge = self.knowledgebase.node_to_implementation_knowledge_dict[
                                searched_node.id
                            ]
                        if (
                            target_knowledge
                            not in queried_knowledge_v2.task_to_similar_task_successful_knowledge[
                                target_task_information
                            ]
                        ):
                            queried_knowledge_v2.task_to_similar_task_successful_knowledge[
                                target_task_information
                            ].append(target_knowledge)

                # finally add embedding related knowledge
                knowledge_base_success_task_list = list(self.knowledgebase.success_task_to_knowledge_dict)

                similarity = calculate_embedding_distance_between_str_list(
                    [target_task_information],
                    knowledge_base_success_task_list,
                )[0]
                similar_indexes = sorted(
                    range(len(similarity)),
                    key=lambda i: similarity[i],
                    reverse=True,
                )
                embedding_similar_successful_knowledge = [
                    self.knowledgebase.success_task_to_knowledge_dict[knowledge_base_success_task_list[index]]
                    for index in similar_indexes
                ]
                for knowledge in embedding_similar_successful_knowledge:
                    if (
                        knowledge
                        not in queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information]
                    ):
                        queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information].append(
                            knowledge
                        )

                if knowledge_sampler > 0:
                    queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information] = [
                        knowledge
                        for knowledge in queried_knowledge_v2.task_to_similar_task_successful_knowledge[
                            target_task_information
                        ]
                        if random.uniform(0, 1) <= knowledge_sampler
                    ]

                # Make sure no less than half of the knowledge are from GT
                queried_knowledge_list = queried_knowledge_v2.task_to_similar_task_successful_knowledge[
                    target_task_information
                ]
                queried_from_gt_knowledge_list = [
                    knowledge
                    for knowledge in queried_knowledge_list
                    if knowledge.feedback is not None and knowledge.feedback.final_decision_based_on_gt == True
                ]
                queried_without_gt_knowledge_list = [
                    knowledge
                    for knowledge in queried_knowledge_list
                    if knowledge.feedback is not None and knowledge.feedback.final_decision_based_on_gt == False
                ]
                queried_from_gt_knowledge_count = max(
                    min((v2_query_component_limit // 2 + 1), len(queried_from_gt_knowledge_list)),
                    v2_query_component_limit - len(queried_without_gt_knowledge_list),
                )
                queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information] = (
                    queried_from_gt_knowledge_list[:queried_from_gt_knowledge_count]
                    + queried_without_gt_knowledge_list[: v2_query_component_limit - queried_from_gt_knowledge_count]
                )

        return queried_knowledge_v2

    def error_query(
        self,
        evo: EvolvableSubjects,
        queried_knowledge_v2: CoSTEERQueriedKnowledgeV2,
        v2_query_error_limit: int = 5,
        knowledge_sampler: float = 1.0,
    ) -> CoSTEERQueriedKnowledge | None:
        for task_index, target_task in enumerate(evo.sub_tasks):
            target_task_information = target_task.get_task_information()
            queried_knowledge_v2.task_to_similar_error_successful_knowledge[target_task_information] = []
            if (
                target_task_information in self.knowledgebase.success_task_to_knowledge_dict
                or target_task_information in queried_knowledge_v2.failed_task_info_set
            ):
                queried_knowledge_v2.task_to_similar_error_successful_knowledge[target_task_information] = []
            else:
                queried_knowledge_v2.task_to_similar_error_successful_knowledge[target_task_information] = []
                if (
                    target_task_information in self.knowledgebase.working_trace_error_analysis
                    and len(self.knowledgebase.working_trace_error_analysis[target_task_information]) > 0
                    and len(queried_knowledge_v2.task_to_former_failed_traces[target_task_information]) > 0
                ):
                    queried_last_trace = queried_knowledge_v2.task_to_former_failed_traces[target_task_information][0][
                        -1
                    ]
                    target_index = self.knowledgebase.working_trace_knowledge[target_task_information].index(
                        queried_last_trace,
                    )
                    last_knowledge_error_analysis_result = self.knowledgebase.working_trace_error_analysis[
                        target_task_information
                    ][target_index]
                else:
                    last_knowledge_error_analysis_result = []

                error_nodes = []
                for error_node in last_knowledge_error_analysis_result:
                    if not isinstance(error_node, UndirectedNode):
                        error_node = self.knowledgebase.graph_get_node_by_content(content=error_node)
                        if error_node is None:
                            continue
                    error_nodes.append(error_node)

                if len(error_nodes) > 1:
                    task_trace_node_list = self.knowledgebase.graph_query_by_intersection(
                        error_nodes,
                        constraint_labels=["task_trace"],
                        output_intersection_origin=True,
                    )
                    single_error_constraint = (v2_query_error_limit // len(error_nodes)) + 1
                else:
                    task_trace_node_list = []
                    single_error_constraint = v2_query_error_limit
                for error_node in error_nodes:
                    # Reverse iterate, a trade-off with intersection search
                    count = 0
                    for task_trace_node in self.knowledgebase.graph_query_by_node(
                        node=error_node,
                        step=1,
                        constraint_labels=["task_trace"],
                        block=True,
                    )[::-1]:
                        if task_trace_node not in task_trace_node_list:
                            task_trace_node_list.append([[error_node], task_trace_node])
                            count += 1
                        if count >= single_error_constraint:
                            break

                # for error_node in last_knowledge_error_analysis_result:
                #     if not isinstance(error_node, UndirectedNode):
                #         error_node = self.knowledgebase.graph_get_node_by_content(content=error_node)
                #         if error_node is None:
                #             continue
                #     for searched_node in self.knowledgebase.graph_query_by_node(
                #         node=error_node,
                #         step=1,
                #         constraint_labels=["task_trace"],
                #         block=True,
                #     ):
                #         if searched_node not in [node[0] for node in task_trace_node_list]:
                #             task_trace_node_list.append((searched_node, error_node.content))

                same_error_success_knowledge_pair_list = []
                same_error_success_node_set = set()
                for error_node_list, trace_node in task_trace_node_list:
                    for searched_trace_success_node in self.knowledgebase.graph_query_by_node(
                        node=trace_node,
                        step=50,
                        constraint_labels=[
                            "task_trace",
                            "task_success_implement",
                            "task_description",
                        ],
                        block=True,
                    ):
                        if (
                            searched_trace_success_node not in same_error_success_node_set
                            and searched_trace_success_node.label == "task_success_implement"
                        ):
                            same_error_success_node_set.add(searched_trace_success_node)

                            trace_knowledge = self.knowledgebase.node_to_implementation_knowledge_dict[trace_node.id]
                            success_knowledge = self.knowledgebase.node_to_implementation_knowledge_dict[
                                searched_trace_success_node.id
                            ]
                            error_content = ""
                            for index, error_node in enumerate(error_node_list):
                                error_content += f"{index+1}. {error_node.content}; "
                            same_error_success_knowledge_pair_list.append(
                                (
                                    error_content,
                                    (trace_knowledge, success_knowledge),
                                ),
                            )

                if knowledge_sampler > 0:
                    same_error_success_knowledge_pair_list = [
                        knowledge
                        for knowledge in same_error_success_knowledge_pair_list
                        if random.uniform(0, 1) <= knowledge_sampler
                    ]

                same_error_success_knowledge_pair_list = same_error_success_knowledge_pair_list[:v2_query_error_limit]
                queried_knowledge_v2.task_to_similar_error_successful_knowledge[target_task_information] = (
                    same_error_success_knowledge_pair_list
                )

        return queried_knowledge_v2


class CoSTEERKnowledgeBaseV2(EvolvingKnowledgeBase):
    def __init__(self, init_component_list=None, path: str | Path = None) -> None:
        """
        Load knowledge, offer brief information of knowledge and common handle interfaces
        """
        self.graph: UndirectedGraph = UndirectedGraph(Path.cwd() / "graph.pkl")
        logger.info(f"Knowledge Graph loaded, size={self.graph.size()}")

        if init_component_list:
            for component in init_component_list:
                exist_node = self.graph.get_node_by_content(content=component)
                node = exist_node if exist_node else UndirectedNode(content=component, label="component")
                self.graph.add_nodes(node=node, neighbors=[])

        # A dict containing all working trace until they fail or succeed
        self.working_trace_knowledge = {}

        # A dict containing error analysis each step aligned with working trace
        self.working_trace_error_analysis = {}

        # Add already success task
        self.success_task_to_knowledge_dict = {}

        # key:node_id(for task trace and success implement), value:knowledge instance(aka 'CoSTEERKnowledge')
        self.node_to_implementation_knowledge_dict = {}

        # store the task description to component nodes
        self.task_to_component_nodes = {}

    def get_all_nodes_by_label(self, label: str) -> list[UndirectedNode]:
        return self.graph.get_all_nodes_by_label(label)

    def update_success_task(
        self,
        success_task_info: str,
    ):  # Transfer the success tasks' working trace to knowledge storage & graph
        success_task_trace = self.working_trace_knowledge[success_task_info]
        success_task_error_analysis_record = (
            self.working_trace_error_analysis[success_task_info]
            if success_task_info in self.working_trace_error_analysis
            else []
        )
        task_des_node = UndirectedNode(content=success_task_info, label="task_description")
        self.graph.add_nodes(
            node=task_des_node,
            neighbors=self.task_to_component_nodes[success_task_info],
        )  # 1st version, we assume that all component nodes are given
        for index, trace_unit in enumerate(success_task_trace):  # every unit: single_knowledge
            neighbor_nodes = [task_des_node]
            if index != len(success_task_trace) - 1:
                trace_node = UndirectedNode(
                    content=trace_unit.get_implementation_and_feedback_str(),
                    label="task_trace",
                )
                self.node_to_implementation_knowledge_dict[trace_node.id] = trace_unit
                for node_index, error_node in enumerate(success_task_error_analysis_record[index]):
                    if type(error_node).__name__ == "str":
                        queried_node = self.graph.get_node_by_content(content=error_node)
                        if queried_node is None:
                            new_error_node = UndirectedNode(content=error_node, label="error")
                            self.graph.add_node(node=new_error_node)
                            success_task_error_analysis_record[index][node_index] = new_error_node
                        else:
                            success_task_error_analysis_record[index][node_index] = queried_node
                neighbor_nodes.extend(success_task_error_analysis_record[index])
                self.graph.add_nodes(node=trace_node, neighbors=neighbor_nodes)
            else:
                success_node = UndirectedNode(
                    content=trace_unit.get_implementation_and_feedback_str(),
                    label="task_success_implement",
                )
                self.graph.add_nodes(node=success_node, neighbors=neighbor_nodes)
                self.node_to_implementation_knowledge_dict[success_node.id] = trace_unit

    def query(self):
        pass

    def graph_get_node_by_content(self, content: str) -> UndirectedNode:
        return self.graph.get_node_by_content(content=content)

    def graph_query_by_content(
        self,
        content: Union[str, list[str]],
        topk_k: int = 5,
        step: int = 1,
        constraint_labels: list[str] = None,
        constraint_node: UndirectedNode = None,
        similarity_threshold: float = 0.0,
        constraint_distance: float = 0,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        search graph by content similarity and connection relationship, return empty list if nodes' chain without node
        near to constraint_node

        Parameters
        ----------
        constraint_distance
        content
        topk_k: the upper number of output for each query, if the number of fit nodes is less than topk_k, return all fit nodes's content
        step
        constraint_labels
        constraint_node
        similarity_threshold
        block: despite the start node, the search can only flow through the constraint_label type nodes

        Returns
        -------

        """

        return self.graph.query_by_content(
            content=content,
            topk_k=topk_k,
            step=step,
            constraint_labels=constraint_labels,
            constraint_node=constraint_node,
            similarity_threshold=similarity_threshold,
            constraint_distance=constraint_distance,
            block=block,
        )

    def graph_query_by_node(
        self,
        node: UndirectedNode,
        step: int = 1,
        constraint_labels: list[str] = None,
        constraint_node: UndirectedNode = None,
        constraint_distance: float = 0,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        search graph by connection, return empty list if nodes' chain without node near to constraint_node
        Parameters
        ----------
        node : start node
        step : the max steps will be searched
        constraint_labels : the labels of output nodes
        constraint_node : the node that the output nodes must connect to
        constraint_distance : the max distance between output nodes and constraint_node
        block: despite the start node, the search can only flow through the constraint_label type nodes

        Returns
        -------
        A list of nodes

        """
        nodes = self.graph.query_by_node(
            node=node,
            step=step,
            constraint_labels=constraint_labels,
            constraint_node=constraint_node,
            constraint_distance=constraint_distance,
            block=block,
        )
        return nodes

    def graph_query_by_intersection(
        self,
        nodes: list[UndirectedNode],
        steps: int = 1,
        constraint_labels: list[str] = None,
        output_intersection_origin: bool = False,
    ) -> list[UndirectedNode] | list[list[list[UndirectedNode], UndirectedNode]]:
        """
        search graph by node intersection, node intersected by a higher frequency has a prior order in the list
        Parameters
        ----------
        nodes : node list
        step : the max steps will be searched
        constraint_labels : the labels of output nodes
        output_intersection_origin: output the list that contains the node which form this intersection node

        Returns
        -------
        A list of nodes

        """
        node_count = len(nodes)
        assert node_count >= 2, "nodes length must >=2"
        intersection_node_list = []
        if output_intersection_origin:
            origin_list = []
        for k in range(node_count, 1, -1):
            possible_combinations = combinations(nodes, k)
            for possible_combination in possible_combinations:
                node_list = list(possible_combination)
                intersection_node_list.extend(
                    self.graph.get_nodes_intersection(node_list, steps=steps, constraint_labels=constraint_labels),
                )
                if output_intersection_origin:
                    for _ in range(len(intersection_node_list)):
                        origin_list.append(node_list)
        intersection_node_list_sort_by_freq = []
        for index, node in enumerate(intersection_node_list):
            if node not in intersection_node_list_sort_by_freq:
                if output_intersection_origin:
                    intersection_node_list_sort_by_freq.append([origin_list[index], node])
                else:
                    intersection_node_list_sort_by_freq.append(node)

        return intersection_node_list_sort_by_freq
