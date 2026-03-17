# WebShop 任务

## 目标
训练模型在 WebShop 电商网站环境中获得更高的购物任务成功率。这是一个**交互式**任务：模型需要在网页环境中多步决策（rollout），根据用户指令搜索并购买匹配的产品。

## 环境概述

WebShop 是一个模拟电商网站环境，包含 118 万真实产品和用户指令。Agent 需要根据文本指令完成购物任务。

环境有 4 种页面状态：
- **search** - 搜索页面，包含搜索框
- **results** - 搜索结果页，列出匹配的产品
- **item** - 产品详情页
- **item-detail** - 产品详细信息页

## 动作空间

Agent 的动作是文本格式，有两种类型：

1. **搜索**: `search[query]` - 在搜索页面使用
   - 示例：`search[red running shoes]`

2. **选择**: `choose[option]` - 根据当前页面选择选项
   - `choose[Back to Search]` - 返回搜索页
   - `choose[Next >]` / `choose[< Prev]` - 翻页
   - `choose[Product Title]` - 选择产品
   - `choose[Option]` - 选择颜色/尺寸等变体
   - `choose[Description]` - 查看详情
   - `choose[Buy Now]` - 购买产品

## Rollout 流程

每轮购物任务的交互循环：

```python
# 初始化
obs, info = env.reset(idx=instruction_idx)  # 获取初始观察（搜索页面）

done = False
for step in range(max_steps):
    # 1. 模型根据指令、历史、当前观察生成动作
    action = model(instruction, history, obs)
    
    # 2. 环境执行动作
    obs, reward, done, info = env.step(action)
    
    # 3. 记录历史
    history.append((action, obs))
    
    if done:
        break

# reward: 最终奖励 (0-1)，反映产品匹配程度
```

**一个 rollout 示例**：

```
指令: "I'm looking for a quick-release replacement fitness strap band; 
       it should match my chic teal fitbit, and price lower than 40.00 dollars"

Step 1: 观察: "WebShop [SEP] Search [SEP]"
        动作: "search[quick-release fitness strap band teal fitbit]"

Step 2: 观察: "WebShop [SEP] Results [SEP] [Back to Search] [Next >] 
               [Teal Silicone Sport Band for Fitbit... $12.99] 
               [Quick Release Nylon Band Teal... $15.99]..."
        动作: "choose[Teal Silicone Sport Band for Fitbit Charge 2, Large, $12.99]"

Step 3: 观察: "WebShop [SEP] Item [SEP] Teal Silicone Sport Band... 
               [Buy Now] [Back to Search] [Description] [Size Large] [Size Small]"
        动作: "choose[Buy Now]"

Step 4: 观察: "WebShop [SEP] Episode finished [SEP] reward = 0.95"
        结果: 任务完成，奖励 0.95（高匹配度）
```

## 观测格式

环境返回的观测是文本格式：

```
WebShop [SEP] {Page Type} [SEP] {Content}
```

- `WebShop` - 固定前缀
- `{Page Type}` - 页面类型：Search / Results / Item
- `{Content}` - 页面内容，包括可用选项

## 评测指标

- **成功率** = 成功购买匹配产品的比例（reward >= 0.5 视为成功）
- **平均奖励** = 所有任务的平均奖励值（0-1），基于产品类型、属性、价格匹配度计算

## 参考代码

环境交互和评测的完整实现见 `eval.py`。
