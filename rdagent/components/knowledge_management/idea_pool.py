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
            "idea": "A concise label summarizing the core concept of this idea (e.g., feature engineering, hyperparameter tuning, dimensionality reduction, ensemble learning, feature selection).",
            "component": "A specific component the idea is focusing on. Must be one of ('DataLoadSpec', 'FeatureEng', 'Model', 'Ensemble', 'Workflow')."
            "method": "A specific method used in this idea (e.g., apply Synthetic Minority Oversampling Technique (SMOTE) to handle imbalanced datasets).The target component to implement the idea can be identified(e.g., feature engineering, hyperparameter tuning, dimensionality reduction, ensemble learning, feature selection). It should be unambiguously implemented in code level(e.g. ensemble with linear regression on validation data with MSE loss). ",
            "context": "An example of how the notebook incorporate  this idea in their solution (e.g. the notebook combines prediction from XGBoost and Randomforest to improve the performance).",
            "hypothesis": {
                "problem": "The nature of the problem (e.g., definition, objective, constraints).",
                "data": "The nature of the data (e.g., size, quality, distribution).",
                "method": "The characteristics of the method (e.g., dependencies, assumptions, strengths).",
                "reason": "A comprehensive analysis of why this method works well in this scenario. You can list multiple  requirements about the scenarios. Here are some examples, the scenario contains patterns in the time-series; there are a lot of outliers in the data; the number of data sample is small; the data is very noisy",
        }
        """
        self.idea = raw_knowledge["idea"]
        self.component = raw_knowledge["component"]
        self.method = raw_knowledge["method"]
        self.context = raw_knowledge["context"]
        self.hypothesis = raw_knowledge["hypothesis"].copy()

    
    def knowledge(self):
        return f"Idea: {self.idea}\nComponent: {self.component}\nMethod: {self.method}\nContext: {self.context}"


class DSKnowledgeGraph(UndirectedGraph):
    def __init__(self, path: str | Path | None = None, idea_cache_path: str | Path | None = None):
        self.idea_pool = {}
        super().__init__(path)
        if idea_cache_path:
            self.build_idea_pool(idea_cache_path)


    def build_idea_pool(self, cache_path: str | Path | None = None):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for i, raw_knowledge in tqdm(enumerate(data), desc="Building Knowledge Graph from Ideas"):
            try:
                idea = Idea(raw_knowledge)

                idea_text = idea.idea
                data = idea.hypothesis["data"]
                problem = idea.hypothesis["problem"]

                idea_node = UndirectedNode(idea_text, "IDEA")
                data_node = UndirectedNode(data, "DATA")
                problem_node = UndirectedNode(problem, "PROBLEM")

                self.add_nodes(idea_node, [data_node, problem_node])
                self.idea_pool[idea_node.id] = idea
            except Exception as e:
                print(f"Fail to add idea {i} to knowledge base due to error: {e}")

    
if __name__ == "__main__":
    output_path = "git_ignore_folder/ds_graph_idea_pool_v1.pkl"
    cache_path = "scripts/exp/researcher/output_dir/idea_pool/test.json"
    idea_pool = DSKnowledgeGraph(path=output_path, idea_cache_path=cache_path)
    idea_pool.dump()