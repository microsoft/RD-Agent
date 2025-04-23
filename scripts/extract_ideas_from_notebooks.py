import json
import os
from pathlib import Path
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.app.data_science.loop import DataScienceRDLoop
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.kaggle_crawler import download_data
from rdagent.utils.agent.tpl import T
import re
from tqdm import tqdm
from typing import Dict

def generate_competition_description(competition):
    DS_RD_SETTING.competition = competition
    if DS_RD_SETTING.competition:

        if DS_RD_SETTING.scen.endswith("KaggleScen"):
            download_data(competition=DS_RD_SETTING.competition, settings=DS_RD_SETTING)
        else:
            return
    ds_loop = DataScienceRDLoop(DS_RD_SETTING)
    return ds_loop.trace.scen.get_scenario_all_desc(eda_output=None)

def process_single_notebook(competition, notebook_path, output_path):
    # prepare
    component_desc = "\n".join(
        [
            f"[{key}] {value}"
            for key, value in T("scenarios.data_science.share:component_description").template.items()
        ]
    )
    scenario_desc = generate_competition_description(competition)
    with open(notebook_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # extract ideas
    system_prompt = T(".prompts:extract_ideas.system").r(
        component_desc=component_desc,
    )
    user_prompt = T(".prompts:extract_ideas.user").r(
        scenario_desc=scenario_desc,
        sota_code=code,
    )
    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )
    match = re.search(r'\[(?:[^\[\]]|\[.*\])*\]', response)
    if match:
        resp_dict = json.loads(match.group(0))

    # save to json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resp_dict, f, indent=2)

# %%
if __name__ == "__main__":
    notebook_folder = "/data/share_folder/mle_lite_medal_solutions"
    output_folder = "/data/userdata/v-xuminrui/Knowledge"

    for competition in tqdm(os.listdir(notebook_folder), desc="Processing Notebooks"):
        competition_name = competition.split(".py")[0]

        process_single_notebook(
            competition_name,
            os.path.join(notebook_folder, competition),
            os.path.join(output_folder, f"{competition_name}.json"),
        )
