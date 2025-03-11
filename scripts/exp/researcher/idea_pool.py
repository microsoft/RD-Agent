import json
import random
import re
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
from jinja2 import Environment, StrictUndefined
from tqdm import tqdm

from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import (
    APIBackend,
    calculate_embedding_distance_between_str_list,
)
import pickle
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode,
)


class Idea:
    def __init__(self, raw_knowledge: Dict) -> None:
        """
        {
        "idea": "A concise label summarizing the core concept of this idea (e.g., feature engineering, hyperparameter tuning, dimensionality reduction, ensemble learning, feature selection).",
        "method": "A specific method used in this idea (e.g., apply Synthetic Minority Oversampling Technique (SMOTE) to handle imbalanced datasets).The target component to implement the idea can be identified(e.g., feature engineering, hyperparameter tuning, dimensionality reduction, ensemble learning, feature selection). It should be unambiguously implemented in code level(e.g. ensemble with linear regression on validation data with MSE loss). ",
        "context": "An example of how the notebook incorporate  this idea in their solution (e.g. the notebook combines prediction from XGBoost and Randomforest to improve the performance).",
        "component": "An specific component of ['DataLoadSpec', 'FeatureEng', 'Model', 'Ensemble', 'Workflow'] that this idea focus on.",
        "hypothesis": {
            "problem": "The nature of the problem (e.g., definition, objective, constraints).",
            "data": "The nature of the data (e.g., size, quality, distribution).",
            "method": "The characteristics of the method (e.g., dependencies, assumptions, strengths).",
            "reason": "A comprehensive analysis of why this method works well in this scenario. You can list multiple  requirements about the scenarios. Here are some examples, the scenario contains patterns in the time-series; there are a lot of outliers in the data; the number of data sample is small; the data is very noisy",
        }
        """
        self.idea = raw_knowledge["idea"]
        self.method = raw_knowledge["method"]
        self.context = raw_knowledge["context"]
        self.hypothesis = raw_knowledge["hypothesis"].copy()
        self.knowledge = self.knowledge()
        self.status = True # indicate whether this idea has been retrieved before
        self.component = raw_knowledge.get("component", None)

    def format_JSON(self) -> str:
        idea_dict = {
            "idea": self.idea,
            "method": self.method,
            "context": self.context,
            "component": self.component,
            "hypothesis": {
                "problem": self.hypothesis["problem"],
                "data": self.hypothesis["data"],
                "method": self.hypothesis["method"],
                "reason": self.hypothesis["reason"],
            },
        }
        return json.dumps(idea_dict)

    def format_text(self, idx) -> str:
        idea_text = f"""## Idea {idx}: {self.idea}
**Overview of Idea**  
In the context of {self.idea}, the idea uses {self.method} to address a specific challenge in the machine learning workflow.
For example, an example scenario of incorporating this idea is that {self.context}

**Hypothesis of the Idea**
The hypothesis supporting this idea is structured as follows: 
- **Problem Nature:**  
  The problem being addressed is: {self.hypothesis["problem"]}  
- **Data Characteristics:**  
  This idea assumes the data has the following properties: {self.hypothesis["data"]}  
- **Method Characteristics:**  
  The method works effectively under these conditions: {self.hypothesis["method"]}  
- **Reasoning:**  
  This method is effective in this scenario because: {self.hypothesis["reason"]}"""
        return idea_text

    def knowledge(self) -> str:
        knowledge = f"""For data: {self.hypothesis['problem']}{self.hypothesis['data']}
For method: {self.hypothesis['method']}
This is because {self.hypothesis['reason']}"""

        return knowledge

    def apply_template(self) -> str:
        pass


class Idea_Pool:
    def __init__(self, threshold=0.8, cache_path=None) -> None:
        """
        Initialize the Idea_Pool class.

        Args:
            threshold (float): Similarity threshold for idea comparison.
            cache_path (str): Path to a cache file for loading existing ideas.
        """
        self.threshold = threshold
        self.idea_pool = []
        if cache_path is not None:
            self.load_from_cache(cache_path)

    def update_old_idea(self, old_idea_idx, new_idea) -> None:
        """
        If a new idea is similar to previous idea, we need to decide whether to update the previous idea.
                1) Update the tags.
                2) Update the reasons.
        3) Refine the method.
        """
        pass

    def load_from_cache(self, cache_path) -> None:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.idea_pool = [Idea(raw_knowledge=idea) for idea in tqdm(data, desc="Building Idea Pool from Ideas")]

    def save_to_cache(self, cache_path) -> None:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.idea_pool, f, indent=4)

    def retrieve_based_on_LLM(self, new_idea, k=8):
        """
        Based on the new idea, retrieve the same ideas from Idea Pool.
        """
        new_idea_text = "# New Idea\n" + str({"idea": new_idea["idea"], "method": new_idea["method"]})
        all_decision = []
        for i in range(0, len(self.idea_pool), k):
            old_ideas = self.idea_pool[i : i + k]
            old_idea_text = "\n".join(
                [
                    f"# Old Idea {idx + 1}\n{str({'idea': idea['idea'], 'method': idea['method']})}"
                    for idx, idea in enumerate(old_ideas)
                ]
            )

            prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")
            sys_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["retrieve_same_ideas"]["system"])
                .render()
            )

            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["retrieve_same_ideas"]["user"])
                .render(new_idea=new_idea_text, old_ideas=old_idea_text)
            )

            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                json_mode=False,
            )

            try:
                k_decision = json.loads(response)
            except:
                match = re.search(r"\[(?:[^\[\]]|\[.*\])*\]", response)
                k_decision = json.loads(match.group(0)) if match else None

            all_decision.extend(k_decision)
        return all_decision

    def retrieve_and_refine_based_on_LLM(self, new_idea, k=5):
        new_idea_text = "# New Idea\n" + str(new_idea.format_JSON())
        for i in range(0, len(self.idea_pool), k):
            old_ideas = self.idea_pool[i : i + k]
            old_idea_text = "\n".join(
                [f"# Old Idea {idx + 1}\n{str(idea.format_JSON())}" for idx, idea in enumerate(old_ideas)]
            )

            prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")
            sys_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["retrieve_and_refine_same_ideas"]["system"])
                .render()
            )

            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["retrieve_and_refine_same_ideas"]["user"])
                .render(new_idea=new_idea_text, old_ideas=old_idea_text)
            )

            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                json_mode=False,
            )
            print(response)

    def add_new_idea(self, new_idea: Idea) -> None:
        """
        Currently, we skip the retrieve and refine steps as we sample based on hypothesis instead of method.
        """
        self.idea_pool.append(new_idea)

    def calculate_sim_matrix(self, source, target):
        sim_matrix = calculate_embedding_distance_between_str_list(
            source_str_list=source, target_str_list=target
        )  # [s,t]

        return np.array(sim_matrix).flatten()  # [s,t] -> [s*t]

    def random_sample(self, k=5):
        random_indices = random.sample(range(len(self.idea_pool)), k)
        sampled_ideas = [self.idea_pool[i] for i in random_indices]
        return sampled_ideas

    def sample(self, solution, k=5):
        """
        Sample top-k ideas based on similarity to the given solution.

        Args:
            solution (str or List[str]): Solution to compare against.
            k (int): Number of top ideas to retrieve.

        Returns:
            Tuple[List[Idea], np.ndarray]: Top ideas and their similarity scores.
        """
        if not self.idea_pool:
            raise ValueError("Idea pool is empty. Add ideas before sampling.")

        unused_idea_pool = [idea for idea in self.idea_pool if idea.status]  # only consider the unused ideas
        # source = [idea.knowledge for idea in unused_idea_pool]  # [s]
        source = [f"The characteristic of the data is {idea.hypothesis['data']}.\nThis is because {idea.hypothesis['reason']}" for idea in unused_idea_pool]
        target = [solution] if isinstance(solution, str) else solution  # [t]
        sim_matrix = self.calculate_sim_matrix(source, target)  # [s*t]

        # get top-k indices
        max_indices = np.argpartition(sim_matrix, -k)[-k:]
        max_indices = max_indices[np.argsort(sim_matrix[max_indices])][::-1]
        max_values = sim_matrix[max_indices]

        # retrieve ideas
        top_ideas = []
        for i in max_indices:
            map_idx = i % len(unused_idea_pool)
            top_ideas.append(unused_idea_pool[map_idx])
            unused_idea_pool[map_idx].status = False
        return top_ideas, max_values


class IdeaKnowledgeGraph(UndirectedGraph):
    def __init__(self, path=None, idea_pool: Idea_Pool = None) -> None:
        super().__init__(path)
        self.build_from_idea_pool(idea_pool)

    def build_from_idea_pool(self, idea_pool: Idea_Pool):
        if idea_pool is not None: 
            for i, idea in tqdm(enumerate(idea_pool.idea_pool), desc="Building Knowledge Graph from Idea Pool"):
                try: 
                    data = idea.hypothesis["data"]
                    problem = idea.hypothesis["problem"]
                    reason = idea.hypothesis["reason"]
                    idea = f"Idea: {idea.idea}\nMethod: {idea.method}\nContext: {idea.context}"

                    data_node = UndirectedNode(data, "DATA")
                    problem_node = UndirectedNode(problem, "PROBLEM")
                    reason_node = UndirectedNode(reason, "METHOD")
                    idea_node = UndirectedNode(idea, "IDEA")
                    self.add_nodes(idea_node, [data_node, problem_node, reason_node])
                except Exception as e:
                    print(f"Fail to add idea {i} to knowledge base due to error: {e}")


if __name__ == "__main__":
    idea_pool = Idea_Pool(cache_path="scripts/exp/researcher/output_dir/idea_pool/test.json")
    kg = IdeaKnowledgeGraph(path="git_ignore_folder/ds_graph_idea_pool_v1.pkl", idea_pool=idea_pool)
    kg.dump()