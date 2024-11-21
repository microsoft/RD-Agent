import random

from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.core.evaluation import Scenario
from rdagent.log import rdagent_logger as logger


def random_select(
    to_be_finished_task_index: list,
    evo: EvolvingItem,
    selected_num: int,
    queried_knowledge: CoSTEERQueriedKnowledge,
    scen: Scenario,
):

    to_be_finished_task_index = random.sample(
        to_be_finished_task_index,
        selected_num,
    )

    logger.info(f"The random selection is: {to_be_finished_task_index}")
    return to_be_finished_task_index
