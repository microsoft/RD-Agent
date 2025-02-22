import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from jinja2 import Environment, StrictUndefined
from tqdm import tqdm

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.prompts import Prompts
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.experiment.scenario import KGScenario

PROMPT_DICT = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class KGKnowledgeGraph(UndirectedGraph):
    def __init__(self, path: str | Path | None, scenario: KGScenario | None) -> None:
        super().__init__(path)
        if path is not None and Path(path).exists():
            self.load()
            self.path = Path(path).parent / (
                datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S") + "_kaggle_kb.pkl"
            )
        else:
            documents = []
            print(Path(KAGGLE_IMPLEMENT_SETTING.domain_knowledge_path))
            for file_path in (Path(KAGGLE_IMPLEMENT_SETTING.domain_knowledge_path)).rglob("*.case"):
                with open(file_path, "r") as f:
                    documents.append(f.read())
            self.load_from_documents(documents=documents, scenario=scenario)
            self.dump()

    def add_document(self, document_content: str, scenario: KGScenario | None) -> None:
        self.load_from_documents([document_content], scenario)
        self.dump()  # Each valid experiment will overwrite this file once again.

    def analyze_one_document(self, document_content: str, scenario: KGScenario | None) -> list:
        session_system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(PROMPT_DICT["extract_knowledge_graph_from_document"]["system"])
            .render(scenario=scenario.get_scenario_all_desc() if scenario is not None else "")
        )

        session = APIBackend().build_chat_session(
            session_system_prompt=session_system_prompt,
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(PROMPT_DICT["extract_knowledge_graph_from_document"]["user"])
            .render(document_content=document_content)
        )
        knowledge_list = []
        for _ in range(10):
            response = session.build_chat_completion(user_prompt=user_prompt, json_mode=True)
            knowledge = json.loads(response)
            knowledge_list.append(knowledge)
            user_prompt = "Continue from the last step please. Don't extract the same knowledge again."
        return knowledge_list

    def load_from_documents(self, documents: List[str], scenario: KGScenario | None) -> None:
        knowledge_list_list = multiprocessing_wrapper(
            [
                (
                    self.analyze_one_document,
                    (
                        document_content,
                        scenario,
                    ),
                )
                for document_content in documents
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )
        node_pairs = []
        node_list = []
        for knowledge_list in tqdm(knowledge_list_list):
            for knowledge in knowledge_list:
                if knowledge == {}:
                    break
                competition = knowledge.get("competition", "")

                competition_node = UndirectedNode(
                    content=(
                        "General knowledge not related to any competition"
                        if (competition == "" or competition == "N/A")
                        else competition
                    ),
                    label="competition",
                )
                node_list.append(competition_node)

                for action in ["hypothesis", "experiments", "code", "conclusion"]:
                    if action == "hypothesis":
                        if isinstance(knowledge.get("hypothesis", ""), str) and knowledge.get("hypothesis", "") in [
                            "N/A",
                            "",
                        ]:
                            break
                        label = knowledge[action]["type"]
                    else:
                        label = action
                    content = str(knowledge.get(action, ""))
                    if content == "" or content == "N/A":
                        continue
                    node = UndirectedNode(content=content, label=label)
                    node_list.append(node)
                    node_pairs.append((node, competition_node))

        node_list = self.batch_embedding(node_list)
        for node_pair in node_pairs:
            self.add_node(node_pair[0], node_pair[1])


if __name__ == "__main__":
    graph = KGKnowledgeGraph(path="git_ignore_folder/kg_graph.pkl", scenario=None)
