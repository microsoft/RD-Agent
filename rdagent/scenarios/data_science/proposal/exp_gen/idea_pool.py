import json
from pathlib import Path
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode, # TODO: add appendix attribute to node
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from tqdm import tqdm
from typing import Dict, List


class Idea:
    def __init__(self, raw_knowledge: Dict) -> None:
        """
        {
            "idea": "A concise label summarizing the core concept of this idea.",
            "method": "A specific method used in this idea, described in a general and implementable way (e.g., 'applied a stacking ensemble method to combine predictions from multiple base models'). Avoid mentioning specific models or dataset-specific details to ensure better generalization",
            "context": "A detailed example of how the notebook implements this idea (e.g., 'the notebook used XGBoost, Random Forest, and LightGBM as base models and logistic regression as the meta-model').",
            "hypothesis": {
                "scenario_problem": "The nature of problem the idea addresses, described without referencing the method itself (e.g., 'a classification problem with complex decision boundaries').",
                "feedback_problem": "The characteristics of the data (e.g., imbalance, high dimensionality, collinearity, outliers, missing data, skewed distribution, time-based pattern, etc.) that justify the use of this method.",
            }
        }
        """
        # TODO: add competition name -> avoid using self-generated ideas
        # TODO: align Scenario and Feedback problem (for key and label)
        self.competition = None
        self.idea = raw_knowledge["idea"]
        self.method = raw_knowledge["method"]
        self.context = raw_knowledge["context"]
        self.hypothesis = raw_knowledge["hypothesis"].copy()

    def to_JSON(self) -> Dict:
        return {
            "competition": self.competition,
            "idea": self.idea,
            "method": self.method,
            "context": self.context,
            "hypothesis": self.hypothesis,
        }
    
class DSKnowledgeGraph(UndirectedGraph):
    def __init__(self, path: str | Path, idea_path: str | Path | None = None):
        self.idea_status_dict = {}
        super().__init__(path)
        if idea_path:
            self.build_idea_pool(idea_path)

    def build_idea_pool(self, idea_path: str | Path):
        with open(idea_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for i, raw_idea in tqdm(enumerate(data), desc="Building Knowledge Graph from Ideas"):
            try:
                idea = Idea(raw_idea)

                idea_name = idea.idea
                data = idea.hypothesis["data"]
                problem = idea.hypothesis["problem"]

                idea_node = UndirectedNode(content=idea_name, label="IDEA", appendix=idea.to_JSON())
                sp_node = UndirectedNode(content=data, label="SCENARIO_PROBLEM")
                fp_node = UndirectedNode(content=problem, label="FEEDBACK_PROBLEM")

                self.add_nodes(idea_node, [sp_node, fp_node])
            
            except Exception as e:
                print(f"The {i}-th idea process failed due to error {e}")

    def sample_ideas(
        self,
        problems: Dict,
        scenario_desc: str,
        exp_and_feedback_list_desc: str,
        sota_exp_desc: str,
    ) -> Dict:
        # sample ideas by cosine similarity
        text = ""
        for i, (key, value) in enumerate(problems.items()):
            sampled_ideas, sampled_ideas_id = [], []
            sampled_nodes = self.semantic_search(
                node=value['problem'], 
                topk_k=5, 
                constraint_labels=[value['label']]
            )
            for node in sampled_nodes:
                idea_node = self.get_nodes_within_steps(
                    start_node=node, 
                    steps=1, 
                    constraint_labels='IDEA'
                )[0]
                if not idea_node.id in self.idea_status_dict:
                    idea = idea_node.appendix
                    sampled_ideas.append(idea)
                    sampled_ideas_id.append(idea_node.id)
            
            text += f"# Problem Name {i+1}: {key}\n"
            text += f"- Problem Description: {value['problem']}\n"
            for j, idea in enumerate(sampled_ideas):
                text += f"## Idea {j+1}\n"
                text += f"- Idea Name: {idea.idea}\n"
                text += f"- Idea Method: {idea.method}\n"
                text += f"- Idea Context: {idea.context}\n\n"
            text += "\n\n"

        # select ideas by LLM
        sys_prompt = T(".prompts_v2:idea_sample.system").r(
            idea_spec=T(".prompts_v2:specification.idea").r(),
            idea_output_format=T(".prompts_v2:output_format.idea").r(),
        )
        user_prompt = T(".prompts_v2:idea_sample.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_and_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            problem_ideas=text,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, List[int]]],
        )
        resp_dict = json.loads(response)

        # update problems with selected ideas
        selected_ideas = []
        for key, value in resp_dict.items():
            for i in value["selected_ideas"]:
                selected_ideas.append(sampled_ideas[i])
                self.idea_status_dict.append(sampled_ideas_id[i])
            problems[key]["ideas"] = selected_ideas
        
        return problems
