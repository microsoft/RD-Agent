import json
from pathlib import Path
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode,  # TODO: add appendix attribute to node
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from tqdm import tqdm
from typing import Dict, List
from rdagent.log import rdagent_logger as logger


class DSIdea:
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
        self.competition = raw_knowledge.get("competition", None)
        self.idea = raw_knowledge["idea"]
        self.method = raw_knowledge["method"]
        self.context = raw_knowledge["context"]
        self.hypothesis = raw_knowledge["hypothesis"].copy()

    def __str__(self) -> str:
        return json.dumps(
            {
                "competition": self.competition,
                "idea": self.idea,
                "method": self.method,
                "context": self.context,
                "hypothesis": self.hypothesis,
            }
        )


class DSKnowledgeBase(UndirectedGraph):
    def __init__(self, path: str | Path | None = None, idea_pool_json_path: str | Path | None = None):
        super().__init__(path)
        self.used_idea_id_set = set()
        if idea_pool_json_path is not None:
            self.build_idea_pool(idea_pool_json_path)
        self.dump()

    def build_idea_pool(self, idea_pool_json_path: str | Path):
        if len(self.vector_base.vector_df) > 0:
            logger.warning("Knowledge graph is not empty, please clear it first. Ignore reading from json file.")
            return
        with open(idea_pool_json_path, "r", encoding="utf-8") as f:
            idea_pool_dict = json.load(f)

        all_nodes = []
        add_nodes_list = []
        for i, raw_idea in tqdm(enumerate(idea_pool_dict), desc="Building Knowledge Graph from Ideas"):
            try:
                idea = DSIdea(raw_idea)

                idea_name = idea.idea
                data = idea.hypothesis["scenario_problem"]
                problem = idea.hypothesis["feedback_problem"]

                idea_node = UndirectedNode(content=idea_name, label="IDEA", appendix=str(idea))
                sp_node = UndirectedNode(content=data, label="SCENARIO_PROBLEM")
                fp_node = UndirectedNode(content=problem, label="FEEDBACK_PROBLEM")
                all_nodes.extend([idea_node, sp_node, fp_node])
                add_nodes_list.append((idea_node, [sp_node, fp_node]))
            except Exception as e:
                print(f"The {i}-th idea process failed due to error {e}")

        self.batch_embedding(all_nodes)
        for add_nodes_pair in add_nodes_list:
            self.add_nodes(add_nodes_pair[0], add_nodes_pair[1])

    def sample_ideas(
        self,
        problems: Dict,
        scenario_desc: str,
        exp_feedback_list_desc: str,
        sota_exp_desc: str,
    ) -> Dict:
        # sample ideas by cosine similarity
        text = ""
        problem_to_sampled_idea_node_id = {}
        for i, (problem_name, problem_dict) in enumerate(problems.items()):
            sampled_nodes = self.semantic_search(
                node=problem_dict["problem"], topk_k=5, constraint_labels=[problem_dict["label"]]
            )

            text += f"# Problem Name {i+1}: {problem_name}\n"
            text += f"- Problem Description: {problem_dict['problem']}\n"
            problem_to_sampled_idea_node_id[problem_name] = []
            for node in sampled_nodes:
                idea_node = self.get_nodes_within_steps(start_node=node, steps=1, constraint_labels="IDEA")[0]
                if not idea_node.id in self.used_idea_id_set:

                    idea = DSIdea(raw_knowledge=json.loads(idea_node.appendix))
                    problem_to_sampled_idea_node_id[problem_name].append(idea_node)
                    text += f"## Idea {len(problem_to_sampled_idea_node_id[problem_name])}\n"
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
            exp_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            problem_ideas=text,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, int],
        )
        resp_dict = json.loads(response)

        # update problems with selected ideas
        # selected_ideas = []
        # for key, value in resp_dict.items():
        #     for i in value["selected_ideas"]:
        #         selected_ideas.append(sampled_ideas[i])
        #         self.used_idea_id_set.append(sampled_ideas_id[i])
        #     problems[key]["ideas"] = selected_ideas

        return problems
