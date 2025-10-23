import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rdagent.app.data_science.conf import DS_RD_SETTING
# =====================
# Reward Model Wrapper
# =====================
class RewardModelInference(nn.Module):
    def __init__(self, base_model_path, adapter_path, reward_head_path, calib_path=None, use_bf16=False):
        super().__init__()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        # hidden size
        hs = getattr(self.model.config, "hidden_size",
                     getattr(self.model.config, "n_embd",
                     getattr(self.model.config, "d_model", None)))
        if hs is None:
            hs = self.model.transformer.wte.embedding_dim

        # reward head
        self.reward_head = nn.Linear(hs, 1)

        state_dict = torch.load(reward_head_path, map_location="cpu", weights_only=True)
        self.reward_head.load_state_dict(state_dict)
        self.reward_head = self.reward_head.to(dtype=self.model.dtype)
        self.reward_head.eval()

        # load calibration parameters
        self.calib = {"a": 1.0, "b": 0.0, "tau": 1.0}
        if calib_path and os.path.exists(calib_path):
            with open(calib_path, "r", encoding="utf-8") as f:
                self.calib = json.load(f)

        # ✅ 打印调试信息，确认精度一致
        print(f"[INFO] Model dtype: {self.model.dtype}, Reward head dtype: {next(self.reward_head.parameters()).dtype}")

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
        pooled = last_hidden.gather(1, idx).squeeze(1)
        reward = self.reward_head(pooled).squeeze(-1)
        return reward

    def compute_reward(self, texts, tokenizer, system_prompt=None, device="cuda"):
        if system_prompt is None:
            system_prompt = (
                "You are an experienced data science competition judge. "
                "Evaluate the quality, effectiveness, and innovation of the proposed solutions."
            )

        inputs = [f"{system_prompt} Solution: {t}{tokenizer.eos_token}" for t in texts]

        enc = tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt"
        ).to(device)

        rewards = self.forward(enc["input_ids"], enc["attention_mask"])
        # Apply calibration
        rewards = self.calib["a"] * rewards + self.calib["b"]
        return rewards.cpu().exp().tolist()

# =====================
# Example Usage
# =====================

