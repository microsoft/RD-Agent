"""
This is the preliminary version of the APE (Automated Prompt Engineering)
"""

import pickle
from pathlib import Path

from rdagent.log.conf import LOG_SETTINGS


def get_llm_qa(file_path):
    data_flt = []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print(len(data))
        for item in data:
            if "debug_llm" in item["tag"]:
                data_flt.append(item)
    return data_flt


# Example usage
# use
file_path = Path(LOG_SETTINGS.trace_path) / "debug_llm.pkl"
llm_qa = get_llm_qa(file_path)
print(len(llm_qa))

print(llm_qa[0])

# Initialize APE backend
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

api = APIBackend()

# Analyze test data and generate improved prompts
for qa in llm_qa:
    # Generate system prompt for APE
    system_prompt = T(".prompts:ape.system").r()

    # Generate user prompt with context from LLM QA
    user_prompt = T(".prompts:ape.user").r(
        system=qa["obj"].get("system", ""), user=qa["obj"]["user"], answer=qa["obj"]["resp"]
    )
    analysis_result = api.build_messages_and_create_chat_completion(
        system_prompt=system_prompt, user_prompt=user_prompt
    )
    print(f"█" * 60)
    yes = input("Do you want to continue? (y/n)")
