import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.app.data_science.conf import DS_RD_SETTING


# =====================
# Reward Model Wrapper
# =====================
class RewardModelInference(nn.Module):
    def __init__(self, base_model_name, adapter_path, reward_head_path, device="cuda"):
        super().__init__()
        self.device = device
        self.base = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.base = PeftModel.from_pretrained(self.base, adapter_path)
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable()
        if hasattr(self.base.config, "use_cache"):
            self.base.config.use_cache = False
        hs = getattr(self.base.config, "hidden_size",
                     getattr(self.base.config, "n_embd",
                     getattr(self.base.config, "d_model", None)))
        if hs is None:
            hs = self.base.get_input_embeddings().embedding_dim

        self.reward_head = nn.Linear(hs, 1).to(device)
        self.reward_head.load_state_dict(torch.load(reward_head_path, map_location=device))

    @staticmethod
    def pool_last_nonpad(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        lengths = attn_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
        return last_hidden.gather(1, idx).squeeze(1)

    def forward(self, input_ids, attention_mask):
        out = self.base(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
            use_cache=False
        )
        last_hidden = out.hidden_states[-1]
        pooled = self.pool_last_nonpad(last_hidden, attention_mask)
        reward = self.reward_head(pooled).squeeze(-1)
        return reward

    def compute_reward(self, texts, tokenizer,comp_description, system_prompt=None, device="cuda"):
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif not hasattr(self, "system_prompt"):
            self.system_prompt = (
                "You are a senior data science competition judge and solution expert.\n"
                "Your task is to evaluate the quality, reasoning progression, and innovation of hypothesis chains.\n"
                "A hypothesis chain shows iterative improvement of solutions.\n"
                "You should assess:\n"
                "1) reasoning correctness and consistency across steps,\n"
                "2) improvement and refinement through the chain,\n"
                "3) final hypothesis quality and practicality.\n"
                "Be strict and fair. Provide expert-level insight."
            )

        inputs = []
        for s in texts:
            prompt = (
                f"{self.system_prompt}\n\n"
                f"Competition description:\n{comp_description}\n\n"
                "Hypothesis Chain (each step separated by '->'):\n"
                f"{s}\n\n"
                "<think>\n"
                "Analyze the evolution of hypotheses, step-by-step, identifying strengths, weaknesses, and logical progression.\n"
                "Focus on clarity, correctness, and improvement.\n"
                "Make sure to consider the chain direction from earliest to latest.\n"
                "</think>\n\n"
                "Final Evaluation:\n"
            )

            inputs.append(prompt)

        enc = tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=DS_RD_SETTING.max_length,
            return_tensors="pt"
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        rewards = self.forward(enc["input_ids"], enc["attention_mask"])

        return torch.exp(rewards).cpu().tolist()