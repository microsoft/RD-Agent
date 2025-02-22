## Usage

This folder implements a knowledge base using RAG based on Kaggle competitions. 
It allows you to store Kaggle competition experiences into the knowledge base, as well as store experimental experiences from RD-Agent.

1. First, generate a knowledge base (in JSON format) by running the `main` function in `extract_knowledge.py`.
2. Then, create a vector base in `vector_base.py` and save it.
3. Finally, add the field `KG_RAG_PATH="xxx.pkl"` (the path to the saved vector base) in your `.env` file.