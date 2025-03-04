import os, json
from pathlib import Path
from scripts.exp.researcher.idea_pool import Idea, Idea_Pool

script_dir = Path(__file__).parent
idea_path = script_dir / "scripts/exp/researcher/output_dir/extracted_ideas"
idea_files = [file for file in os.listdir(idea_path) if file.endswith('.json')]


pool = Idea_Pool()
for file in idea_files: 
    with open(f"{idea_path}/{file}", "r") as f:
        data = json.load(f)
    for idea in data:
        pool.add_new_idea(idea)

output_path = script_dir / "scripts/exp/researcher/output_dir/idea_pool/test.json"
pool.save_to_cache(str(output_path))