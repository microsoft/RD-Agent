import json
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
from rdagent.oai.llm_utils import APIBackend

class WorkflowMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: WorkflowTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
    ) -> dict[str, str]:
        competition_info = self.scen.competition_descriptions
        
        system_prompt = T(".prompts:workflow_coder.system").r()
        user_prompt = T(".prompts:workflow_coder.user").r(
            competition_info=competition_info,
        )

        data_loader_code = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
            )
        )["code"]
        
        return
    
