import json
from pathlib import Path

def merge_jsons():
    merged_dict = {}
    for idea_file in Path(r"/data/userdata/v-xuminrui/Knowledge/archive").iterdir():
        competition_name = idea_file.stem
        merged_dict[competition_name] = json.load(idea_file.open())
    json.dump(
        merged_dict,
        open(r"/data/userdata/v-xuminrui/Knowledge/archive/merged_ideas.json", "w"),
        indent=2,
    )

if __name__ == "__main__":
    merge_jsons()