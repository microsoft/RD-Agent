# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
import json
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, DSDockerConf

DIRNAME = Path(__file__).absolute().resolve().parent

DataLoaderEvalFeedback = CoSTEERSingleFeedback


class DataLoaderCoSTEEREvaluator(CoSTEEREvaluator):

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: CoSTEERQueriedKnowledgeV2 = None,
        **kwargs,
    ) -> DataLoaderEvalFeedback:

        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return DataLoaderEvalFeedback(
                execution="This task has failed too many times, skip implementation.",
                return_checking="This task has failed too many times, skip implementation.",
                code="This task has failed too many times, skip implementation.",
                final_decision=False,
            )

        ds_docker_conf = DSDockerConf()
        ds_docker_conf.extra_volumes = {
            f"{DS_RD_SETTING.local_data_path}/sample/{self.scen.competition}": "/kaggle/input"
        }
        de = DockerEnv(conf=ds_docker_conf)

        # TODO: do we need to clean the generated temporary content?
        fname = "data_loader_test.py"
        test_code = (DIRNAME / "eval_tests" / "data_loader_test.txt").read_text()
        implementation.inject_files(**{fname: test_code})
        stdout = implementation.execute(env=de, entry=f"python {fname}")

        system_prompt = T(".prompts:data_loader_eval.system").r(
            task_desc=target_task.get_task_information(),
            test_code=test_code,
            code=implementation.file_dict["load_data.py"],
        )
        user_prompt = T(".prompts:data_loader_eval.user").r(stdout=stdout)

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
        return DataLoaderEvalFeedback(**json.loads(resp))
