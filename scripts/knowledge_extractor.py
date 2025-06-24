import json
import os
import re
from pathlib import Path
from typing import Dict, List

import nbformat
from nbconvert import MarkdownExporter

from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

LOCAL_DATA_PATH = "/data/userdata/v-xuminrui/knowledge/knowledge_v1"

"""
📁 knowledge_v1
├── 📁 <competition_name>
├──── 🗒️ <competition_name>.md # competition description
├──── 📁 discussion
│        🗒️ discussion_<id>_<idx>.md
├──── 📁 notebook
│        📁 notebook_<id>_<idx>
│           🗒️ <notebook_name>.ipynb
"""


def notebook_to_markdown(notebook):
    with open(notebook, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    exporter = MarkdownExporter()
    (body, resources) = exporter.from_notebook_node(notebook)

    return body


def notebook_to_knowledge(competition_desc: str, discussion: str, notebook: str):
    sys_prompt = T(".prompts:extract_specific_knowledge.system").r(
        component_desc=T(".prompts:describe.component_description").r(),
        output_format=T(".prompts:output_format.specific_knowledge").r(),
    )
    user_prompt = T(".prompts:extract_specific_knowledge.user").r(
        competition_desc=competition_desc,
        notebook=notebook,
        discussion=discussion,
    )
    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=True,
        json_target_type=Dict[str, Dict[str, str]],
    )
    return json.loads(response)


def process_single_competition(competition_name: str):
    competition_path = f"{LOCAL_DATA_PATH}/{competition_name}"
    with open(f"{competition_path}/{competition_name}.md", encoding="utf-8") as file:
        competition_desc = file.read()

    discussion_files = os.listdir(f"{competition_path}/discussion")
    notebook_files = os.listdir(f"{competition_path}/notebook")

    for discussion in discussion_files:
        match = re.match(r"discussion_(\d+_\d+)\.md", discussion)
        if not match:
            continue
        idx = match.group(1)

        with open(f"{competition_path}/discussion/{discussion}", encoding="utf-8") as file:
            discussion_content = file.read()

        notebook = f"{competition_path}/notebook/notebook_{idx}"
        if not os.path.isdir(notebook):
            continue

        notebook_file = [f for f in os.listdir(notebook) if f.endswith(".ipynb")][0]
        notebook_content = notebook_to_markdown(f"{notebook}/{notebook_file}")

        knowledge = notebook_to_knowledge(
            competition_desc=competition_desc,
            notebook=notebook_content,
            discussion=discussion_content,
        )

        existing_knowledge = {}
        save_path = f"{LOCAL_DATA_PATH}/knowledge.json"
        if Path(save_path).exists():
            with open(save_path, "r", encoding="utf-8") as f:
                existing_knowledge = json.load(f)

        existing_knowledge.update(knowledge)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(existing_knowledge, f, indent=2, ensure_ascii=False)


process_single_competition("feedback-prize-english-language-learning")
