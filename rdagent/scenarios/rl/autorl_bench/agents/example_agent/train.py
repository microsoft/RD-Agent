"""
GRPO Training Loop
"""
import json
import os
import re
import time

import requests
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def extract_answer(text):
    match = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except:
            pass
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except:
            pass
    return None


def load_data(file_path, ratio=1.0):
    records = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            prompt = f"Solve this math problem step by step. Put your final answer after ####.\n\nQuestion: {item['question']}\n\nSolution:"
            records.append({"prompt": prompt, "question": item["question"], "answer": item["answer"]})
    if ratio < 1.0:
        n = max(10, int(len(records) * ratio))
        records = records[:n]
    return records


def gsm8k_reward_func(completions, answer, **kwargs):
    rewards = []
    for completion, gold_answer in zip(completions, answer):
        pred = extract_answer(completion)
        gold = extract_answer(gold_answer)
        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def submit_for_grading(grading_url: str, model_path: str) -> dict | None:
    if not grading_url:
        return None
    try:
        resp = requests.post(f"{grading_url}/submit", json={"model_path": model_path}, timeout=600)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  Grading error: {e}")
    return None


def main():
    MODEL_PATH = os.environ.get("MODEL_PATH")
    DATA_PATH = os.environ.get("DATA_PATH")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/autorl_output")
    GRADING_SERVER_URL = os.environ.get("GRADING_SERVER_URL", "")
    TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.05"))
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
    
    if not MODEL_PATH or not DATA_PATH:
        raise ValueError("MODEL_PATH and DATA_PATH required")

    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    train_file = f"{DATA_PATH}/train.jsonl"
    train_data = load_data(train_file, TRAIN_RATIO)
    print(f"Train samples: {len(train_data)}")
    dataset = Dataset.from_list([{"prompt": d["prompt"], "answer": d["answer"]} for d in train_data])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    # 第一个 epoch 使用原始模型，后续 epoch 使用上一个 checkpoint
    current_model_path = MODEL_PATH

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")

        config = GRPOConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=4,       # 小 batch 避免 OOM
            gradient_accumulation_steps=16,      # 梯度累积保持有效batch=64
            learning_rate=1e-5,
            max_completion_length=256,
            num_generations=4,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            bf16=True,
        )

        # 直接传模型路径，让 GRPOTrainer 自己管理模型加载
        # 避免 vLLM colocate 模式下模型被加载两次导致 OOM
        trainer = GRPOTrainer(
            model=current_model_path,
            reward_funcs=gsm8k_reward_func,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        checkpoint_dir = f"{OUTPUT_DIR}/checkpoint-epoch{epoch + 1}"
        trainer.save_model(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # 下一个 epoch 从这个 checkpoint 继续训练
        current_model_path = checkpoint_dir

        result = submit_for_grading(GRADING_SERVER_URL, checkpoint_dir)
        if result:
            print(f"  Score: {result.get('score')}")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    submit_for_grading(GRADING_SERVER_URL, OUTPUT_DIR)
    print(f"\nDone! Total: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
