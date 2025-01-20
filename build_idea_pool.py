import os, json
from scripts.exp.researcher.idea_pool import Idea, Idea_Pool

idea_path = "scripts/exp/researcher/cache/idea"
idea_files = [file for file in os.listdir(idea_path) if file.endswith('.json')]

max_num = 50
count = 0
pool = Idea_Pool()
for file in idea_files: 
    with open(f"{idea_path}/{file}", "r") as f:
        data = json.load(f)
    for idea in data:
        count += 1
        pool.add_new_idea(idea)
        if count > max_num:
            break
    if count > max_num:
        break

pool.save_to_cache("scripts/exp/researcher/cache/idea_pool/test.json")