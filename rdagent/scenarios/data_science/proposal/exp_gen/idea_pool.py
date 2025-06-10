import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from rdagent.components.knowledge_management.graph import (
    UndirectedNode,  # TODO: add appendix attribute to node
)
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
)
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class DSIdea:
    def __init__(self, raw_knowledge: Dict | str) -> None:
        """
        {
            "idea": "A concise label for the idea.",
            "problem": "The scenario problem that the idea addresses, described without referencing the method itself.",
            "method": "A specific method used in this idea, described in a general and implementable way (e.g., 'applied a stacking ensemble method to combine predictions from multiple base models'). Avoid mentioning dataset-specific details to ensure better generalization",
            "context": "A detailed example of how the notebook implements this idea (e.g., 'the notebook used XGBoost, Random Forest, and LightGBM as base models and logistic regression as the meta-model').",
        }
        """
        if isinstance(raw_knowledge, str):
            raw_knowledge = json.loads(raw_knowledge)
    
        self.idea = raw_knowledge.get("idea", None)
        self.competition = raw_knowledge.get("competition", None)
        self.problem = raw_knowledge.get("problem", None)
        self.method = raw_knowledge.get("method", None)
        self.context = raw_knowledge.get("context", None)

    def __str__(self) -> str:
        return json.dumps(
            {
                "idea": self.idea,
                "competition": self.competition,
                "method": self.method,
                "context": self.context,
                "problem": self.problem,
            }
        )

    def to_formatted_str(self) -> str:
        return f"Idea Name: {self.idea}\nIdea Method: {self.method}\nIdea Context: {self.context}"


class DSKnowledgeBase(UndirectedGraph):
    def __init__(self, path: str | Path | None = None, idea_pool_json_path: str | Path | None = None):
        super().__init__(path)
        self.used_idea_id_set = set()
        if idea_pool_json_path is not None:
            self.build_idea_pool(idea_pool_json_path)
        self.dump()

    def add_idea(self, idea: List[DSIdea] | DSIdea) -> None:
        if not isinstance(idea, list):
            idea_list = [idea]
        else:
            idea_list = idea

        node_list = []
        add_pairs = []
        for one_idea in idea_list:
            idea_name = one_idea.idea
            idea_node = UndirectedNode(content=idea_name, label="IDEA", appendix=str(one_idea))
            node_list.append(idea_node)

            competition = one_idea.competition
            if competition is not None:
                competition_node = UndirectedNode(content=competition, label="COMPETITION")
                node_list.append(competition_node)
                add_pairs.append((idea_node, [competition_node]))

            problem = one_idea.problem
            if problem is not None:
                problem_node = UndirectedNode(content=problem, label="PROBLEM")
                node_list.append(problem_node)
                add_pairs.append((idea_node, [problem_node]))

        self.batch_embedding(node_list)
        for idea_node, neighbor_list in add_pairs:
            self.add_nodes(idea_node, neighbor_list)

    def build_idea_pool(self, idea_pool_json_path: str | Path):
        if len(self.vector_base.vector_df) > 0:
            logger.warning("Knowledge graph is not empty, please clear it first. Ignore reading from json file.")
            return
        else:
            logger.info(f"Building knowledge graph from idea pool json file: {idea_pool_json_path}")
        with open(idea_pool_json_path, "r", encoding="utf-8") as f:
            idea_pool_dict = json.load(f)

        to_add_ideas = []
        for i, (idea_label, idea_content) in tqdm(enumerate(idea_pool_dict.items()), desc="Building Knowledge Graph from Ideas"):
            idea_content.update({"idea": idea_label})
            try:
                idea = DSIdea(idea_content)
                to_add_ideas.append(idea)
            except Exception as e:
                print(f"The {i}-th idea process failed due to error {e}")
                continue
        self.add_idea(to_add_ideas)

    # TODO: move the cosine similarity sampling methods to a separate module
    def sample_by_similarity():
        pass

    # TODO: add the keyword-based sampling methods to a separate module using BM25
    def sample_by_keywords():
        pass
    
    # TODO: move the LLM-based sampling methods to a separate module
    def sample_by_LLM():
        pass

    def sample_ideas(
        self,
        problems: Dict,
        scenario_desc: str,
        exp_feedback_list_desc: str,
        sota_exp_desc: str,
        competition_desc: str,
    ) -> Dict:
        # sample ideas by cosine similarity
        text = ""
        problem_to_sampled_idea_node_id = {}
        competition_node = self.get_node_by_content(competition_desc)

        for i, (problem_name, problem_dict) in enumerate(problems.items()):
            sampled_nodes = self.semantic_search(
                node=problem_dict["problem"], constraint_labels=['PROBLEM']
            )

            text += f"Problem Name {i+1}: {problem_name}\n"
            text += f"- Problem Description: {problem_dict['problem']}\n"
            problem_to_sampled_idea_node_id[problem_name] = []
            for node in sampled_nodes:
                idea_node = self.get_nodes_within_steps(start_node=node, steps=1, constraint_labels="IDEA")[0]

                if idea_node.id not in self.used_idea_id_set and (
                    competition_node is None or competition_node not in idea_node.neighbors
                ):
                    idea = DSIdea(raw_knowledge=idea_node.appendix)
                    problem_to_sampled_idea_node_id[problem_name].append(idea_node)
                    text += f"## Idea {len(problem_to_sampled_idea_node_id[problem_name])}\n"
                    text += f"- Idea Name: {idea.idea}\n"
                    text += f"- Idea Method: {idea.method}\n"
                    text += f"- Idea Context: {idea.context}\n\n"
                # if len(problem_to_sampled_idea_node_id[problem_name]) >= 5:
                #     break
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
        for problem_name, picked_id in resp_dict.items():
            if problem_name in problem_to_sampled_idea_node_id and picked_id < len(
                problem_to_sampled_idea_node_id[problem_name]
            ):
                problems[problem_name]["idea"] = problem_to_sampled_idea_node_id[problem_name][picked_id - 1].appendix
                problems[problem_name]["idea_node_id"] = problem_to_sampled_idea_node_id[problem_name][picked_id - 1].id

        return problems

    def update_picked_problem(self, problems: Dict, picked_problem_name: str) -> None:
        picked_id = problems[picked_problem_name].get("idea_node_id", None)
        if picked_id is not None:
            self.used_idea_id_set.add(picked_id)