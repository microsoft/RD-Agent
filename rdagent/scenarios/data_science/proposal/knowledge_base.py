import pickle

from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode,
)
from scripts.exp.researcher.idea_pool import Idea_Pool


class DSKnowledgeGraph(UndirectedGraph):
    def __init__(self, path=None, idea_pool: Idea_Pool = None) -> None:
        super().__init__(path)
        self.load_from_idea_pool(idea_pool)

    def load_from_idea_pool(self, idea_pool: Idea_Pool):
        for idea in idea_pool.idea_pool:
            if idea.target_component is None or idea.target_component == "others":
                continue

            data = idea.hypothesis["data"]
            problem = idea.hypothesis["problem"]
            method = idea.hypothesis["method"] + "\nBecause: " + idea.hypothesis["reason"]

            idea = f"Idea: {idea.idea}\nMethod: {idea.method}\Context: {idea.context}"

            data_node = UndirectedNode(data, "data")
            problem_node = UndirectedNode(problem, "problem")
            method_node = UndirectedNode(method, "method")
            idea_node = UndirectedNode(idea, "idea")
            self.add_nodes(idea_node, [data_node, problem_node, method_node])


if __name__ == "__main__":
    idea_pool = pickle.load(open("git_ignore_folder/idea_pool.pkl", "rb"))
    kg = DSKnowledgeGraph(idea_pool=idea_pool)
    kg.dump("git_ignore_folder/ds_graph_idea_pool_v1.pkl")
