from scripts.exp.researcher.idea_pool import Idea, Idea_Pool
from scripts.exp.researcher.kaggle_crawler import solution_to_feature
import json, re

idea_pool = Idea_Pool(cache_path="scripts/exp/researcher/output_dir/idea_pool/test.json")

with open("scripts/exp/researcher/output_dir/solution/solution_example.txt") as f:
    solution = f.read()

solution_feature = solution_to_feature(solution)
try: 
    features = json.loads(solution_feature)
except: 
    match = re.search(r'\[(?:[^\[\]]|\[.*\])*\]', solution_feature)
    features = json.loads(match.group(0)) if match else None

if features is None: 
    idea, sim = self.idea_pool.sample(solution_feature, k=1)
    idea = idea[0]
else: 
    extracted_features = []   
    for feat in features:
        characteristic, contents = next(iter(feat.items()))
        if contents['Assessment'].lower() == 'no':
            temp = f'''The characteristic of the data is {characteristic}.
    This is because {contents['Reason']}'''
            extracted_features.append(temp)
    idea, sim = idea_pool.sample(extracted_features, k=1)

print(idea)
print(sim)
