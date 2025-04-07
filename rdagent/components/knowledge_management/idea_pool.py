import json
from pathlib import Path
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode,
)
from tqdm import tqdm
from typing import Dict


class Idea:
    def __init__(self, raw_knowledge: Dict) -> None:
        """
        {
            "idea": "A concise label summarizing the core concept of this idea.",
            "method": "A specific method used in this idea, described in a general and implementable way (e.g., 'applied a stacking ensemble method to combine predictions from multiple base models'). Avoid mentioning specific models or dataset-specific details to ensure better generalization",
            "context": "A detailed example of how the notebook implements this idea (e.g., 'the notebook used XGBoost, Random Forest, and LightGBM as base models and logistic regression as the meta-model').",
            "hypothesis": {
                "problem": "The nature of problem the idea addresses, described without referencing the method itself (e.g., 'a classification problem with complex decision boundaries').",
                "data": "The characteristics of the data (e.g., imbalance, high dimensionality, collinearity, outliers, missing data, skewed distribution, time-based pattern, etc.) that justify the use of this method.",
            }
        }
        """
        self.idea = raw_knowledge["idea"]
        self.method = raw_knowledge["method"]
        self.context = raw_knowledge["context"]
        self.hypothesis = raw_knowledge["hypothesis"].copy()

    
class DSKnowledgeGraph(UndirectedGraph):
    def __init__(self, path: str | Path, idea_path: str | Path | None = None):
        self.idea_pool = {}
        super().__init__(path)
        if idea_path:
            self.build_idea_pool(idea_path)

    def build_idea_pool(self, idea_path: str | Path):
        with open(idea_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for i, raw_idea in tqdm(enumerate(data), desc="Building Knowledge Graph from Ideas"):
            try:
                idea = Idea(raw_idea)

                idea_text = idea.idea
                data = idea.hypothesis["data"]
                problem = idea.hypothesis["problem"]

                idea_node = UndirectedNode(idea_text, "IDEA")
                data_node = UndirectedNode(data, "DATA")
                problem_node = UndirectedNode(problem, "PROBLEM")

                self.add_nodes(idea_node, [data_node, problem_node])
                self.idea_pool[idea_node.id] = idea
            
            except Exception as e:
                print(f"The {i}-th idea process failed due to error {e}")

    def sample_ideas(self, problems: Dict) -> Dict:
        pass

    
if __name__ == "__main__":
    output_path = "scripts/exp/researcher/output_dir/idea_pool/ds_graph_idea_pool_v2.pkl"
    cache_path = "scripts/exp/researcher/output_dir/idea_pool/idea_v2.json"
    idea_pool = DSKnowledgeGraph(path=output_path, idea_cache_path=cache_path)
    idea_pool.dump()