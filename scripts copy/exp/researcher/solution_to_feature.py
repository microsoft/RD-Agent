import os
import json
from pathlib import Path
from kaggle_crawler import solution_to_feature


competition_path = "/data/userdata/share/kaggle/playground-series-s3e14.json"
notebook_path = "/data/userdata/v-xuminrui/RD-Agent/scripts/exp/researcher/optiver-trading-at-the-close/naive_solution.txt"


with open("/data/userdata/share/kaggle/playground-series-s3e14.json", "r") as f:
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

solution_to_feature(inputs)