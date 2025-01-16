from typing import List, Dict
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from jinja2 import Environment, StrictUndefined
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import APIBackend, calculate_embedding_distance_between_str_list

class Idea:
    def __init__(self, raw_knowledge: Dict) -> None:
        '''
        {
            "idea": "A concise label summarizing the core concept of this idea.",
            "method": "A specific method used in this idea.",
            "code": "A simplified pseudocode or code snippet to represent coding steps needed to implement the method. Your generated code should inspire other data scientists to easily apply the idea.",
            "hypothesis": {
                "problem": "The nature of the problem.",
                "data": "The nature of the data.",
                "method": "The characteristics of the method.",
                "reason": "A comprehensive analysis of why this method works well in this case."
            }
        }
        '''
        self.idea = raw_knowledge["idea"]
        self.method = raw_knowledge["method"]
        self.context = raw_knowledge["context"]
        self.hypothesis = raw_knowledge["hypothesis"].copy()
        self.knowledge = self.knowledge()

    def format_JSON(self) -> str:
        idea_dict = {
            "idea": self.idea,
            "method": self.method,
            "context": self.context,
            "hypothesis": {
                "problem": self.hypothesis["problem"],
                "data": self.hypothesis["data"],
                "method": self.hypothesis["method"],
                "reason": self.hypothesis["reason"]
            }
        }
        return json.dumps(idea_dict)


    def knowledge(self) -> str: 
        knowledge = f'''Problem Nature: {self.hypothesis['problem']}
Data Nature: {self.hypothesis['data']}
Method Description: {self.method}
Method Characteristics: {self.hypothesis['method']}
Reason for Using Method: {self.hypothesis['reason']}'''
        
        return knowledge

    def apply_template(self) -> str:
        pass

class Idea_Pool:
    def __init__(self, threshold=0.8) -> None:
        self.threshold = threshold
        self.idea_pool = []

    def update_old_idea(self, old_idea_idx, new_idea) -> None: 
        '''
        If a new idea is similar to previous idea, we need to decide whether to update the previous idea.
		1) Update the tags.
		2) Update the reasons.
        3) Refine the method. 
        '''
        pass

    # def retrieve_based_on_LLM(self, new_idea, k=10): 
    #     '''
    #     Based on the new idea, retrieve the similar ideas from Idea Pool.
    #     Two ideas are considered similar if they describe **exactly the same method**, even if they use different wording or examples.
    #     '''
    #     new_idea_text = "# New Idea\n" + str({"idea": new_idea.idea, "method": new_idea.method})
    #     for i in range(0, len(self.idea_pool), k):
    #         old_ideas = self.idea_pool[i:i + k]
    #         old_idea_text = "\n".join([f"# Old Idea {idx + 1}\n{str({"idea": idea.idea, "method": idea.method})}" for idx, idea in enumerate(old_ideas)])

    #         prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")
    #         sys_prompt = (
    #             Environment(undefined=StrictUndefined)
    #             .from_string(prompt_dict["retrieve_same_ideas"]["system"])
    #             .render() 
    #         )

    #         user_prompt = (
    #             Environment(undefined=StrictUndefined)
    #             .from_string(prompt_dict["retrieve_same_ideas"]["user"])
    #             .render(new_idea=new_idea_text, old_ideas=old_idea_text)
    #         )

    #         response = APIBackend().build_messages_and_create_chat_completion(
    #             user_prompt=user_prompt,
    #             system_prompt=sys_prompt,
    #             json_mode=False,
    #         )
    
    def retrieve_and_refine_based_on_LLM(self, new_idea, k=10): 
        new_idea_text = "# New Idea\n" + str(new_idea.format_JSON())
        for i in range(0, len(self.idea_pool), k):
            old_ideas = self.idea_pool[i:i+k]
            old_idea_text = "\n".join([f"# Old Idea {idx + 1}\n{str(idea.format_JSON())}" for idx, idea in enumerate(old_ideas)])

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

    def retrieve_based_on_sim(self, new_idea): 
        '''
        Based on the new idea, retrieve the most similar idea.
        '''
        # Todo (minrui): cache the embedding to avoid repetitive computation. 
        source = [i.knowledge for i in self.idea_pool] # s
        target = [new_idea.knowledge] # t
        sim_matrix = calculate_embedding_distance_between_str_list(
            source_str_list=source, target_str_list=target
        ) # [s, t]

        sim_matrix = np.array(sim_matrix).flatten()
        max_sim = np.max(sim_matrix)
        max_idx = np.argmax(sim_matrix)
        return max_sim, max_idx

        # sim_matrix = np.array(sim_matrix).T # [t, s]
        # top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:][:, ::-1]  # [t, k]
        # top_k_values = np.take_along_axis(sim_matrix, top_k_indices, axis=1)  # [t, k]
        # return top_k_indices, top_k_values

    def add_new_idea(self, new_idea: Idea) -> None:
        '''
        To decide whether a new idea should be added to the pool, 
        we calculate the similarities between the embeddings vectors 
        and add the new idea if all similarities are below the threshold.
        '''
        if len(self.idea_pool) == 0:
            self.idea_pool.append(new_idea)
        else:
            # max_sim, max_idx = self.retrieve_based_on_sim(new_idea)
            # if max_sim >= self.threshold: 
            #     self.update_old_idea(max_idx, new_idea)
            # else: 
            #     self.idea_pool.append(new_idea)  
            self.retrieve_and_refine_based_on_LLM(new_idea)
            self.idea_pool.append(new_idea)