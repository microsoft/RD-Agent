# ALFWorld 任务

## 目标
训练模型在 ALFWorld 文本游戏环境中获得更高的任务成功率。

## 训练方式：纯 SFT（监督微调）

**重要**：训练脚本只能做 SFT（Supervised Fine-Tuning），不能做在线 RL。
- ALFWorld 环境（textworld、alfworld、rdagent）在训练环境中**不可用**，**禁止 import**
- **禁止 pip install** 任何包
- 训练数据是 `react_prompts.json`，包含 36 个 ReAct 风格的专家演示（6 种任务类型 × 6 个示例）
- 用 SFT 让模型学习 ReAct 交互模式：观察 → think/action → 观察 → ...

## 训练数据格式

`react_prompts.json` 是一个 JSON dict，key 是示例名（如 `react_put_0`），value 是完整的交互文本：

```
观察文本（房间描述 + 任务目标）
> think: 推理过程
OK.
> go to cabinet 1
On the cabinet 1, you see a cloth 1...
> take spraybottle 2 from cabinet 2
You pick up the spraybottle 2...
> put spraybottle 2 in/on toilet 1
You put the spraybottle 2 in/on the toilet 1.
```

每个示例是一段连续文本，包含观察和以 `> ` 开头的动作交替出现。将这些文本作为 SFT 训练数据，教模型生成正确的动作响应。

## 评测方式（你不需要实现）

评测由 Grading Server 完成：用 vLLM 加载你训练的模型，在 ALFWorld 环境中做 text completion：
- 输入：few-shot prompt + 当前观察历史 + `\n>`
- 模型生成下一个动作（stop at `\n`）
- 评测指标：任务成功率

## 任务类型（6 种）
1. **pick_and_place** (put): 拿起物品放到指定位置
2. **pick_clean_then_place** (clean): 清洁物品后放到指定位置
3. **pick_heat_then_place** (heat): 加热物品后放到指定位置
4. **pick_cool_then_place** (cool): 冷却物品后放到指定位置
5. **look_at_obj_in_light** (examine): 在灯光下查看物品
6. **pick_two_obj_and_place** (puttwo): 拿起两个物品放到指定位置

## 可用动作空间
- 导航: `go to {object} {id}`
- 拿取: `take {object} {id} from {location} {id}`
- 放置: `put {object} {id} in/on {location} {id}`
- 打开/关闭: `open {object} {id}`, `close {object} {id}`
- 加热/冷却: `heat {object} {id} with microwave {id}`, `cool {object} {id} with fridge {id}`
- 清洁: `clean {object} {id} with sinkbasin {id}`
- 使用: `use {object} {id}`
- 思考: `think: {reasoning}`（不影响环境状态，输出 OK.）

## Requirements
- Use LoRA (PEFT) for parameter-efficient training
- After training, **merge** the LoRA adapter into the base model (`model.merge_and_unload()`) and save the **full merged model** (not just the adapter) to the output directory. The output must contain `config.json` and `model.safetensors` so it can be loaded standalone by vLLM.

## Environment Variables
- `MODEL_PATH`: Path to the base model
- `DATA_PATH`: Path to training data directory
- `OUTPUT_DIR`: Where to save the trained model
- `MAX_STEPS`: Maximum training steps (default: 80)
