import os, json
from pathlib import Path
from scripts.exp.researcher.idea_pool import Idea, Idea_Pool
from scripts.exp.researcher.kaggle_crawler import solution_to_feature

pool = Idea_Pool(cache_path="/data/userdata/v-xhong/ds_researcher/RD-Agent/scripts/exp/researcher/output_dir/idea_pool/test.json")

competition_path = "/data/userdata/share/kaggle/optiver-trading-at-the-close.json"
notebook_path = "/data/userdata/v-xhong/ds_researcher/RD-Agent/scripts/exp/researcher/output_dir/solution/naive_solution.txt"

with open(competition_path, "r") as f:
    data = json.load(f)

with open(notebook_path, 'r', encoding='utf-8') as file:
    text = file.read()

inputs = f'''## Competition Description
{data['Overview']}

## Data Description
{data['Data Description']}

## Solution Notebook
{text}
'''

feature = solution_to_feature(inputs)

top_ideas, max_values = pool.sample(feature)

print(top_ideas, max_values)