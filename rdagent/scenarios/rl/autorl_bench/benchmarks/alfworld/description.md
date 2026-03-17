# ALFWorld 任务

## 目标
训练模型在 ALFWorld 文本游戏环境中获得更高的任务成功率。这是一个**交互式**任务：模型需要在环境中多步决策（rollout），而非一次性生成答案。

## 环境概述
ALFWorld 是一个文本模拟的家庭环境（TextWorld 引擎）。模型扮演 agent，通过文本指令在房间中导航、操作物品来完成任务。

## 任务类型（6 种）
1. **pick_and_place**: 拿起物品放到指定位置
2. **pick_clean_then_place**: 清洁物品后放到指定位置
3. **pick_heat_then_place**: 加热物品后放到指定位置
4. **pick_cool_then_place**: 冷却物品后放到指定位置
5. **look_at_obj_in_light**: 在灯光下查看物品
6. **pick_two_obj_and_place**: 拿起两个物品放到指定位置

## Rollout 流程

每局游戏的交互循环：

```
初始化：ob, info = env.reset()     # 获取初始观察（房间描述 + 任务目标）

循环（每步）：
  action = model(观察历史)           # 模型根据历史生成动作（文本）
  ob, reward, done, info = env.step([action])  # 环境执行动作，返回新观察
  if done:
      break
```

**一个 rollout 示例（pick_and_place）：**
```
任务: "put a pencil in/on shelf."

Step 1:  观察: "You are in the middle of a room. Looking around you, you see a bed 1, a desk 1, a shelf 1..."
         动作: "go to desk 1"
Step 2:  观察: "On the desk 1, you see a pencil 1, a book 2."
         动作: "take pencil 1 from desk 1"
Step 3:  观察: "You pick up the pencil 1 from the desk 1."
         动作: "go to shelf 1"
Step 4:  观察: "You arrive at shelf 1. On the shelf 1, you see nothing."
         动作: "put pencil 1 in/on shelf 1"
Step 5:  观察: "You put the pencil 1 in/on the shelf 1."
         结果: 任务完成
```

## 可用动作空间
Agent 的动作是自由文本，常见动作包括：
- 导航: `go to {object} {id}`（如 `go to desk 1`, `go to fridge 1`）
- 拿取: `take {object} {id} from {location} {id}`
- 放置: `put {object} {id} in/on {location} {id}`
- 打开/关闭: `open {object} {id}`, `close {object} {id}`
- 加热/冷却: `heat {object} {id} with microwave {id}`, `cool {object} {id} with fridge {id}`
- 清洁: `clean {object} {id} with sinkbasin {id}`
- 使用: `use {object} {id}`（如 `use desklamp 1`）
- 思考: `think: {reasoning}`（不影响环境状态）

## 评测指标
- **成功率** = 成功任务数 / 总任务数

## 参考代码
环境交互和评测的完整实现见 `eval.py`。
