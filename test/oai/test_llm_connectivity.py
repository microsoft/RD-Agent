#!/usr/bin/env python3
"""Test LLM connectivity for multiple models in parallel."""
import os
import concurrent.futures

os.environ["OPENAI_API_KEY"] = "sk-1234"
os.environ["OPENAI_API_BASE"] = "http://localhost:4000"

import litellm
litellm.suppress_debug_info = True
from litellm import completion

TIMEOUT = 30

MODELS = [
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "openai/gpt-5.1-chat",
    "openai/gpt-5.2-chat",
    "gpt-4o-mini",
    "o3",
    "o4-mini",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4o",
]

def test_model(model: str) -> tuple:
    try:
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": "Who is the president of the United States?"}],
            drop_params=True,
            timeout=TIMEOUT
        )
        return (model, True, resp.choices[0].message.content)
    except Exception as e:
        return (model, False, str(e))

if __name__ == "__main__":
    print(f"Testing {len(MODELS)} model(s)...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        for model, ok, msg in ex.map(test_model, MODELS):
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {model}: {msg}")
