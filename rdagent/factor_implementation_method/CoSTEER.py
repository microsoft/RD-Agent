from pathlib import Path
from typing import List
from rdagent.core.implementation import FactorGenerator
from rdagent.core.task import FactorTask, TaskImplementation

class CoSTEERFG(FactorGenerator):
    def __init__(
        self,
        max_loops: int = 10,
        selection_method: str = "random",
        selection_ratio: float = 0.5,
        knowledge_base_path: Path = None,
    ) -> None:
        self.max_loops = max_loops
        self.selection_method = selection_method
        self.selection_ratio = selection_ratio
        self.knowledge_base_path = knowledge_base_path
        if self.knowledge_base_path is not None:
            self.knowledge_base_path = Path(knowledge_base_path)

    def generate(self, tasks: List[FactorTask]) -> List[TaskImplementation]:
        pass    

    def collect_feedback(self, feedback_obj: object):
        pass  # in evolving, all previous feedbacks are collected in the knowledge base, so we don't need to collect feedback here
