import json
import re
from pathlib import Path
import pickle
import os
from tqdm import tqdm
class Saver:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def dump(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with path.open("rb") as f:
            return pickle.load(f)


def extract_JSON(text):
    # Try to directly load the text as JSON
    try:
        extracted_text = json.loads(text)
        if isinstance(extracted_text, dict):
            return [extracted_text]
        elif isinstance(extracted_text, list):
            return extracted_text
        else:
            return []
    except json.JSONDecodeError:
        pass
    
    # Try to extract the outermost JSON array
    try:
        match = re.search(r'\[(?:[^\[\]]|\[.*\])*\]', text)
        if match:
            extracted_text = json.loads(match.group(0))
            return extracted_text
    except json.JSONDecodeError:
        pass
    
    # Try to extract the first JSON object
    try:
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            extracted_text = json.loads(match.group(0))
            return [extracted_text]
    except json.JSONDecodeError:
        pass

    return []


def extract_features(raw_features, ftype):
    features = extract_JSON(raw_features)
    extracted_features = []
    if features is not None:
        for feat in features:
            # data features
            if ftype == "data":
                if feat['Assessment'].lower() == 'no':
                    feature, reason = feat['Characteristic'], feat['Reason']
                    extracted_features.append({"label": "DATA", "feature": feature, "reason": reason})
            # problem features
            elif ftype == "problem":
                feature, reason = feat['Problem'], feat['Reason']
                extracted_features.append({"label": "PROBLEM", "feature": feature, "reason": reason})
            else:
                raise NotImplementedError
    return extracted_features


def format_idea(items, component=None):
    idx = 1
    suggested_ideas = ""
    for item in items:
        idea, feature = item
        if component is None or idea.component == component:
            suggested_ideas += f"## Idea {idx}: {idea.idea}\n"
            suggested_ideas += f"Method: {idea.method}\n"
            if component is None: # no specified component
                suggested_ideas += f"Target Component: {idea.component}\n"
            suggested_ideas += f"Target Problem: {feature['feature']} as evidenced by {feature['reason']}\n"
            suggested_ideas += f"Idea Usage Example: {idea.context}\n\n"
            idx += 1
    return suggested_ideas


from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
def solution_to_idea(component_desc, competition_desc, solution) -> str:
    sys_prompt = T(".prompts:solution_to_idea.system").r(component_desc=component_desc)
    user_prompt = T(".prompts:solution_to_idea.user").r(competition_desc=competition_desc, solution=solution)

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    return response


def solution_to_data(competition_desc, solution) -> str:
    sys_prompt = T(".prompts:solution_to_data.system").r()
    user_prompt = T(".prompts:solution_to_data.user").r(competition_desc=competition_desc, solution=solution)

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    return response


def solution_to_problem(competition_desc, solution, feedback) -> str:
    sys_prompt = T(".prompts:solution_to_problem.system").r()
    user_prompt = T(".prompts:solution_to_problem.user").r(competition_desc=competition_desc, solution=solution, feedback=feedback)

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        json_mode=False,
    )
    return response



# import nbformat
# from nbconvert import MarkdownExporter
# from rdagent.scenarios.kaggle.kaggle_crawler import crawl_descriptions, download_notebooks
# def load_description(desc_path):
#     with open(desc_path, "r") as f:
#         data = json.load(f)

#     competition_desc = ""
#     keys = ['Description', 'Overview', 'Data Description']
#     for key in keys:
#         if key in data:
#             competition_desc += f"{data[key]}\n"

#     return competition_desc


# def notebook_to_text(notebook, markdown):
#     with open(notebook, "r", encoding="utf-8") as f:
#         notebook = nbformat.read(f, as_version=4)

#     exporter = MarkdownExporter()
#     (body, resources) = exporter.from_notebook_node(notebook)

#     with open(markdown, "w", encoding="utf-8") as f:
#         f.write(body)
    
#     return body


# def prepare_notebooks(competitions, notebook_path, idea_path, max_num=5):
#     component_desc = "\n".join(
#         [
#             f"[{key}] {value}"
#             for key, value in T("scenarios.data_science.share:component_description").template.items()
#         ]
#     )

#     # download descriptions and notebooks
#     for competition in tqdm(competitions, desc="Downloading descriptions and competitions."):
#         crawl_descriptions(competition, notebook_path)
#         download_notebooks(competition, notebook_path, max_num)
    
#     # extract ideas from notebooks
#     all_ideas = []
#     for competition in tqdm(competitions, desc="Extracting Ideas."):
#         competition_desc = load_description(f"{notebook_path}/{competition}.json")

#         nb_path = f"{notebook_path}/{competition}"
#         for root, dirs, files in os.walk(nb_path):
#             for file in files:
#                 if file.endswith(".ipynb"):
#                     notebook = os.path.join(root, file)
#                     markdown = os.path.join(root, file.replace(".ipynb", ".md"))
#                     if os.path.exists(markdown):
#                         with open(markdown, "r", encoding="utf-8") as f:
#                             solution = f.read()
#                     else:
#                         solution = notebook_to_text(notebook, markdown)
        
#                     extracted_ideas = solution_to_idea(component_desc, competition_desc, solution)
#                     new_ideas = extract_JSON(extracted_ideas)
#                     all_ideas.extend(new_ideas)

#     with open(idea_path, "w", encoding="utf-8") as f:
#         json.dump(all_ideas, f, indent=4)


# # %%
# if __name__ == "__main__":
#     mini_case_cs = [
#         "feedback-prize-english-language-learning",
#         "playground-series-s3e11",
#         "playground-series-s3e14",
#         "spaceship-titanic",
#         "playground-series-s3e18",
#         "playground-series-s3e16",
#         "playground-series-s3e9",
#         "playground-series-s3e25",
#         "playground-series-s3e26",
#         "playground-series-s3e24",
#         "playground-series-s3e23",
#     ]

#     other_cs = [
#         "amp-parkinsons-disease-progression-prediction",
#         "arc-prize-2024",
#         "ariel-data-challenge-2024",
#         "child-mind-institute-detect-sleep-states",
#         "connectx",
#         "contradictory-my-dear-watson",
#         "digit-recognizer",
#         "fathomnet-out-of-sample-detection",
#         "forest-cover-type-prediction",
#         "gan-getting-started",
#         "google-research-identify-contrails-reduce-global-warming",
#         "house-prices-advanced-regression-techniques",
#         "isic-2024-challenge",
#         "leash-BELKA",
#         "llm-20-questions",
#         "nlp-getting-started",
#         "playground-series-s4e1",
#         "playground-series-s4e2",
#         "playground-series-s4e3",
#         "playground-series-s4e4",
#         "playground-series-s4e5",
#         "playground-series-s4e6",
#         "playground-series-s4e7",
#         "playground-series-s4e8",
#         "rsna-2024-lumbar-spine-degenerative-classification",
#         "sf-crime",
#         "store-sales-time-series-forecasting",
#         "titanic",
#         "tpu-getting-started",
#         # scenario competition
#         "covid19-global-forecasting-week-1",
#         "statoil-iceberg-classifier-challenge",
#         "optiver-realized-volatility-prediction",
#         "facebook-v-predicting-check-ins",
#     ]

# # %%

#     all_cs = mini_case_cs + other_cs
#     prepare_notebooks(competitions=all_cs, 
#                       notebook_path="/data/userdata/v-xuminrui/Notebook",
#                       idea_path="/data/userdata/v-xuminrui/RD-Agent/scripts/exp/researcher/output_dir/idea_pool/idea_v3.json")