import re
import tiktoken
from typing import TypedDict
import pandas as pd

import sys

class Message(TypedDict):
    competition_name: str
    location: int
    model: str
    role: str
    tokens: int

log_files = [
    "aerial-cactus-identification.1.stdout",
    "aptos2019-blindness-detection.1.stdout",
    "cassava-leaf-disease-classification.1.stdout",
    "denoising-dirty-documents.1.stdout",
    "detecting-insults-in-social-commentary.1.stdout",
    "dogs-vs-cats-redux-kernels-edition.1.stdout",
    "google-quest-challenge.1.stdout",
    "h-and-m-personalized-fashion-recommendations.1.stdout",
    "herbarium-2020-fgvc7.1.stdout",
    "herbarium-2021-fgvc8.1.stdout",
    "histopathologic-cancer-detection.1.stdout",
    "hotel-id-2021-fgvc8.1.stdout",
    "inaturalist-2019-fgvc6.1.stdout",
    "iwildcam-2019-fgvc6.1.stdout",
    "jigsaw-toxic-comment-classification-challenge.1.stdout",
    "kuzushiji-recognition.1.stdout",
    "leaf-classification.1.stdout",
    "mlsp-2013-birds.1.stdout",
    "nomad2018-predict-transparent-conductors.1.stdout",
    "plant-pathology-2020-fgvc7.1.stdout",
    "plant-pathology-2021-fgvc8.1.stdout",
    "predict-volcanic-eruptions-ingv-oe.1.stdout",
    "random-acts-of-pizza.1.stdout",
    "rsna-miccai-brain-tumor-radiogenomic-classification.1.stdout",
    "seti-breakthrough-listen.1.stdout",
    "spooky-author-identification.1.stdout",
    "stanford-covid-vaccine.1.stdout",
    "tabular-playground-series-dec-2021.1.stdout",
    "text-normalization-challenge-english-language.1.stdout",
    "text-normalization-challenge-russian-language.1.stdout",
    "the-icml-2013-whale-challenge-right-whale-redux.1.stdout",
    "tweet-sentiment-extraction.1.stdout",
    "us-patent-phrase-to-phrase-matching.1.stdout",
    "whale-categorization-playground.1.stdout",
]

log_dir = sys.argv[1]

all_interactions = []

for log_file in log_files:
    print(log_file)
    messages: list[Message] = []
    with open(f"{log_dir}/{log_file}", "r") as f:
        content = f.read()
    for match in re.finditer(r".\[95m.*?Role:.*?\[96m(.*?).\[0m[\s\S]*?\[96m([\s\S]*?).\[0m", content, re.DOTALL):
        messages.append({
            "competition_name": log_file.split(".")[0],
            "location": match.start(0),
            "role": match.group(1).strip(),
            "tokens": len(tiktoken.encoding_for_model("o3").encode(match.group(2).strip())),
            "model": ""
        })
        # break

    for match in re.finditer(r"Using chat model.\[0m (.*?).\[0m[\s\S]*?_create_chat_completion_inner_function.*?assistant:.\[0m.*?\n([\s\S]*?).\[0m", content, re.DOTALL):
        messages.append({
            "competition_name": log_file.split(".")[0],
            "location": match.start(0),
            "role": "assistant",
            "tokens": len(tiktoken.encoding_for_model("o3").encode(match.group(2).strip())),
            "model": match.group(1).strip()
        })
        # break

    messages = sorted(messages, key=lambda x: x["location"])
    assistant_tokens = user_tokens = 0
    model = ""
    interactions = []
    for msg in messages[::-1]:
        if msg["role"] == "assistant":
            assistant_tokens = msg["tokens"]
            model = msg["model"]
        elif msg["role"] == "user":
            user_tokens = msg["tokens"]
        elif msg["role"] == "system" and assistant_tokens > 0:
            interactions.append({"log_offset": msg["location"], "system": msg["tokens"], "user": user_tokens, "system_and_user": msg["tokens"] + user_tokens, "assistant": assistant_tokens, "model": model})
            assistant_tokens = user_tokens = 0

    interactions = interactions[::-1]

    all_interactions += [{"competition": log_file.split(".")[0], **i} for i in interactions]

    # break

pd.DataFrame(all_interactions).to_csv("logs_0523/rdagent_0523_tokens.csv", index=False)

# print(all_interactions)
