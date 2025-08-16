import json

from pydantic import BaseModel, Field

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.core.experiment import FBWorkspace
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class LeakageCheck(BaseModel):

    leakage: list[str] = Field(
        description="A list of data leakage cases. For each case, describe in detail which part of the code causes the leakage and what information is exposed. If there are no leakages, return an empty list."
    )
    decision: bool = Field(
        description="If the code pass the data leakage check. True means data leakages do not happen, False means leakages happens."
    )


class LeakageEvaluator(CoSTEEREvaluator):

    def evaluate(
        self, target_task, implementation: FBWorkspace, gt_implementation=None, queried_knowledge=None, **kwargs
    ):
        system_prompt = T(".prompts:leakage_eval.system").r()

        user_prompt = T(".prompts:leakage_eval.user").r(
            scenario=self.scen.get_scenario_all_desc(),
            code=implementation.all_codes,
        )

        api = APIBackend()
        assert api.supports_response_schema(), "Leakage evaluation requires response schema"
        response = api.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=LeakageCheck,
        )
        leakage_check = LeakageCheck(**json.loads(response))
        if leakage_check.decision:
            return CoSTEERSingleFeedback(
                execution="",
                return_checking="[Data Leakage Check] The results are reliable. No data leakage was detected.",
                code="[Data Leakage Check] The code is reliable. No data leakage was found.",
                final_decision=True,
            )

        code_leakage_msg = T(".prompts:leakage_eval.code_leakage").r(leakage_check=leakage_check)
        return CoSTEERSingleFeedback(
            execution="",
            return_checking="[Data Leakage Detected] The results are not reliable. Data leakage was found.",
            code=code_leakage_msg,
            final_decision=False,
        )
