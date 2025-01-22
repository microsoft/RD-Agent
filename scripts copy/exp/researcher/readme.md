```
export PYTHONPATH=$(pwd):PYTHONPATH

# generate ideas from notebooks
dotenv run -- python scripts/exp/researcher/json_gen.py

# build idea pool
dotenv run -- python build_idea_pool.py
```