import os, json
from scripts.exp.researcher.idea_pool import Idea, Idea_Pool

pool = Idea_Pool(cache_path="scripts/exp/researcher/cache/idea_pool/test.json")

# load solution_to_feature result

feature = ...

pool.sample(feature)