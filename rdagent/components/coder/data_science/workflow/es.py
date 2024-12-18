import json
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
from rdagent.oai.llm_utils import APIBackend
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.core.experiment import FBWorkspace
from rdagent.utils.agent.tpl import T

class WorkflowMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: WorkflowTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
    ) -> dict[str, str]:
        # competition_info = self.scen.competition_descriptions
        
        system_prompt = T(".prompts:workflow_coder.system").r(
            workflow_spec=workspace.code_dict["spec/workflow.md"]
        )
        user_prompt = T(".prompts:workflow_coder.user").r(
            load_data_code=workspace.code_dict["load_data.py"],
            feature_code=workspace.code_dict["feat01.py"],
            model_code=workspace.code_dict["model01.py"],
            ensemble_code=workspace.code_dict["ens.py"],
        )
        data_loader_code = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
            )
        )["code"]
        
        return{
            "main.py": data_loader_code
        }
    
    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """
        Assign the code list to the evolving item.

        The code list is aligned with the evolving item's sub-tasks.
        If a task is not implemented, put a None in the list.
        """
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                # evo.sub_workspace_list[index] = FBWorkspace(target_task=evo.sub_tasks[index])
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_code(**code_list[index])
        return evo

    
