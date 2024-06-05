from finco.llm import APIBackend
from scripts.factor_implementation.share_modules.prompt import (
    FactorImplementationPrompts,
)
from jinja2 import Template
from scripts.factor_implementation.share_modules.conf import FactorImplementSettings
import json
from scripts.factor_implementation.share_modules.utils import get_data_folder_intro
from scripts.factor_implementation.baselines.evolving.evolvable_subjects import (
    FactorImplementationList,
)
def RandomSelect(to_be_finished_task_index, implementation_factors_per_round):
    import random
    to_be_finished_task_index = random.sample(
        to_be_finished_task_index,
        implementation_factors_per_round,
    )
    print("The random selection is:",to_be_finished_task_index)
    return to_be_finished_task_index

def LLMSelect(to_be_finished_task_index, implementation_factors_per_round, evo:FactorImplementationList, former_trace):
    tasks = []
    for i in to_be_finished_task_index:
    # find corresponding former trace for each task
        for t in former_trace:
            # It is the first try if the list is empty, return None.
            if len(former_trace[t]) == 0:
                return None
            if former_trace[t][0].target_task.factor_name == evo.target_factor_tasks[i].factor_name:
                tasks.append((i, evo.target_factor_tasks[i], former_trace[t]))
                break

    system_prompt = Template(
        FactorImplementationPrompts("prompts.yaml")["select_implementable_factor_naive_system"],
    ).render(
        data_info=get_data_folder_intro(),
    )

    session = APIBackend(use_chat_cache=False).build_chat_session(
        session_system_prompt=system_prompt,
    )

    while True:
        user_prompt = (
            Template(
                FactorImplementationPrompts("prompts.yaml")["select_implementable_factor_naive_user"],
            )
            .render(
                factor_num = implementation_factors_per_round,
                target_factor_tasks=tasks,
            )
        )
        if (
            session.build_chat_completion_message_and_calculate_token(
                user_prompt,
            )
            < FactorImplementSettings().chat_token_limit
        ):
            break

    response = session.build_chat_completion(
        user_prompt=user_prompt,
        json_mode=True,
    )
    code = json.loads(response)["selected_factor"]
    return code

def LLMCoTSelect(to_be_finished_task_index, implementation_factors_per_round, evo:FactorImplementationList, former_trace):
    tasks = []
    for i in to_be_finished_task_index:
    # find corresponding former trace for each task
        flag = 0
        if len(former_trace) == 0:
            tasks.append((i, evo.target_factor_tasks[i]))
        else:
            for t in former_trace:
                if former_trace[t][0].target_task.factor_name == evo.target_factor_tasks[i].factor_name:
                    tasks.append((i, evo.target_factor_tasks[i], former_trace[t]))
                    flag = 1
                    break
            # if former trace not exist, but it still in to_be_finished_task_index
            if flag == 0:
                tasks.append((i, evo.target_factor_tasks[i]))
        

    system_prompt = Template(
        FactorImplementationPrompts("prompts.yaml")["select_implementable_factor_CoT_system"],
    ).render(
        data_info=get_data_folder_intro(),
    )

    session = APIBackend(use_chat_cache=False).build_chat_session(
        session_system_prompt=system_prompt,
    )

    while True:
        user_prompt = (
            Template(
                FactorImplementationPrompts("prompts.yaml")["select_implementable_factor_naive_user"],
            )
            .render(
                factor_num = implementation_factors_per_round,
                target_factor_tasks=tasks,
            )
        )
        if (
            session.build_chat_completion_message_and_calculate_token(
                user_prompt,
            )
            < FactorImplementSettings().chat_token_limit
        ):
            break

    response = session.build_chat_completion(
        user_prompt=user_prompt,
        json_mode=True,
    )
    try:
        selection = json.loads(response)["selected_factor"]
        if not isinstance(selection, list):
            return to_be_finished_task_index
        selection_index = [x for x in selection if isinstance(x, int)]
    except:
        return to_be_finished_task_index
    
    return selection_index
