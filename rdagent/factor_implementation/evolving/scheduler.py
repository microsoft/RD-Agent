from rdagent.oai.llm_utils import APIBackend
from jinja2 import Template
import json
from rdagent.factor_implementation.share_modules.factor_implementation_utils import get_data_folder_intro
from rdagent.factor_implementation.evolving.factor import FactorEvovlingItem
from rdagent.core.prompts import Prompts
from rdagent.core.log import RDAgentLog
from rdagent.core.conf import RD_AGENT_SETTINGS
from pathlib import Path

scheduler_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


def RandomSelect(to_be_finished_task_index, implementation_factors_per_round):
    import random

    to_be_finished_task_index = random.sample(
        to_be_finished_task_index,
        implementation_factors_per_round,
    )

    RDAgentLog().info(f"The random selection is: {to_be_finished_task_index}")
    return to_be_finished_task_index


def LLMSelect(to_be_finished_task_index, implementation_factors_per_round, evo: FactorEvovlingItem, former_trace):
    tasks = []
    for i in to_be_finished_task_index:
        # find corresponding former trace for each task
        target_factor_task_information = evo.target_factor_tasks[i].get_factor_information()
        if target_factor_task_information in former_trace:
            tasks.append((i, evo.target_factor_tasks[i], former_trace[target_factor_task_information]))

    system_prompt = Template(
        scheduler_prompts["select_implementable_factor_system"],
    ).render(
        data_info=get_data_folder_intro(),
    )

    session = APIBackend(use_chat_cache=False).build_chat_session(
        session_system_prompt=system_prompt,
    )

    while True:
        user_prompt = Template(
            scheduler_prompts["select_implementable_factor_user"],
        ).render(
            factor_num=implementation_factors_per_round,
            target_factor_tasks=tasks,
        )
        if (
            session.build_chat_completion_message_and_calculate_token(
                user_prompt,
            )
            < RD_AGENT_SETTINGS.chat_token_limit
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
