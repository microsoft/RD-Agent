import json
import re
from pathlib import Path
import pickle
import os
from tqdm import tqdm
from typing import Dict
import nbformat
from nbconvert import MarkdownExporter
from rdagent.scenarios.kaggle.kaggle_crawler import crawl_descriptions, download_notebooks
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T

from bs4 import BeautifulSoup
import html2text
import re

def load_description(desc_path: str) -> str:
    """Load and convert competition description to compact Markdown format.
    
    Args:
        desc_path: Path to JSON file containing crawled HTML descriptions
        
    Returns:
        Compact Markdown formatted as "Section: content" without extra spacing
    """
    # Initialize compact Markdown converter
    md_converter = html2text.HTML2Text()
    md_converter.body_width = 0       # No line wrapping
    md_converter.single_line_break = True  # Single newlines become spaces
    md_converter.ignore_links = False
    md_converter.ignore_images = True
    md_converter.ul_item_mark = '-'   # Simpler list markers
    
    with open(desc_path, "r", encoding='utf-8') as f:
        html_descriptions = json.load(f)
    
    markdown_lines = []
    
    for title, html_content in html_descriptions.items():
        # Clean HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove empty paragraphs and unnecessary divs
        for element in soup.find_all(['p', 'div']):
            if not element.get_text().strip():
                element.decompose()
        
        # Convert to Markdown
        content = md_converter.handle(str(soup))
        
        # Compact processing
        content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 newlines
        content = re.sub(r' +', ' ', content)         # Single spacing
        content = content.replace('\\', '')           # Remove escapes
        content = content.strip()
        
        # Format as compact section
        markdown_lines.append(f"{title}:\n{content}\n")
    
    return '\n'.join(markdown_lines).strip()


def notebook_to_text(notebook, markdown):
    with open(notebook, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    exporter = MarkdownExporter()
    (body, resources) = exporter.from_notebook_node(notebook)

    with open(markdown, "w", encoding="utf-8") as f:
        f.write(body)
    
    return body

def notebook_to_idea(component_desc, competition_desc, solution):
    # sys_prompt = T(".prompts:extract_ideas.system").r()
    # user_prompt = T(".prompts:extract_ideas.user").r(
    #     competition_desc=competition_desc,
    #     solution=solution,
    # )
    sys_prompt = T(".prompts:extract_scenario_problems_and_ideas.system").r()
    user_prompt = T(".prompts:extract_scenario_problems_and_ideas.user").r(
        competition_desc=competition_desc,
        solution=solution,
    )
    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=True,
        json_target_type=Dict[str, Dict[str, str]],
    )
    return json.loads(response)

def prepare_notebooks(competitions, notebook_path, idea_path, max_num=5):
    component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()

    # download descriptions and notebooks
    # for competition in tqdm(competitions, desc="Downloading descriptions and competitions."):
    #     crawl_descriptions(competition, notebook_path)
    #     download_notebooks(competition, notebook_path, max_num)
    
    # extract ideas from notebooks
    for competition in tqdm(competitions, desc="Extracting Ideas"):
        competition_desc = load_description(f"{notebook_path}/{competition}.json")
        print(competition_desc)

        nb_path = f"{notebook_path}/{competition}"
        for root, dirs, files in os.walk(nb_path):
            for file in files:
                if file.endswith(".ipynb"):
                    notebook = os.path.join(root, file)
                    markdown = os.path.join(root, file.replace(".ipynb", ".md"))
                    
                    if os.path.exists(markdown):
                        with open(markdown, "r", encoding="utf-8") as f:
                            solution = f.read()
                    else:
                        solution = notebook_to_text(notebook, markdown)
                    
                    new_ideas = notebook_to_idea(component_desc, competition_desc, solution)
                    
                    if new_ideas:
                        if os.path.exists(idea_path):
                            with open(idea_path, "r", encoding="utf-8") as f:
                                existing_ideas = json.load(f)
                        else:
                            existing_ideas = {}
                        
                        existing_ideas.update(new_ideas)
                        
                        with open(idea_path, "w", encoding="utf-8") as f:
                            json.dump(existing_ideas, f, indent=4)



# %%
if __name__ == "__main__":
    mini_case_cs = [
        "feedback-prize-english-language-learning",
        "playground-series-s3e11",
        "playground-series-s3e14",
        "spaceship-titanic",
        "playground-series-s3e18",
        "playground-series-s3e16",
        "playground-series-s3e9",
        "playground-series-s3e25",
        "playground-series-s3e26",
        "playground-series-s3e24",
        "playground-series-s3e23",
    ]

    other_cs = [
        "amp-parkinsons-disease-progression-prediction",
        "arc-prize-2024",
        "ariel-data-challenge-2024",
        "child-mind-institute-detect-sleep-states",
        "connectx",
        "contradictory-my-dear-watson",
        "digit-recognizer",
        "fathomnet-out-of-sample-detection",
        "forest-cover-type-prediction",
        "gan-getting-started",
        "google-research-identify-contrails-reduce-global-warming",
        "house-prices-advanced-regression-techniques",
        "isic-2024-challenge",
        "leash-BELKA",
        "llm-20-questions",
        "nlp-getting-started",
        "playground-series-s4e1",
        "playground-series-s4e2",
        "playground-series-s4e3",
        "playground-series-s4e4",
        "playground-series-s4e5",
        "playground-series-s4e6",
        "playground-series-s4e7",
        "playground-series-s4e8",
        "rsna-2024-lumbar-spine-degenerative-classification",
        "sf-crime",
        "store-sales-time-series-forecasting",
        "titanic",
        "tpu-getting-started",
        # scenario competition
        "covid19-global-forecasting-week-1",
        "statoil-iceberg-classifier-challenge",
        "optiver-realized-volatility-prediction",
        "facebook-v-predicting-check-ins",
    ]

# %%

    all_cs = mini_case_cs + other_cs
    prepare_notebooks(competitions=all_cs, 
                      notebook_path="/data/userdata/v-xuminrui/Notebook",
                      idea_path="/data/userdata/v-xuminrui/Knowledge/idea_v4.json")