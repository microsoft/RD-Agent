
```task

我们要构建一个用于 RL-post-training（强化学习后训练）的 benchmark 和 agent 系统。

可以大量参考 `rdagent/scenarios/finetune` 的内容。

我们从 benchmark 开始。

最终的 benchmark 代码将实现在 rdagent/scenarios/rl/eval

待办事项清单：
- [ ] 构建一个 example workspace（workspace 是 agent 系统生成的解决方案）并在 docker 环境中运行
  - 为评测构建专用环境，并编写测试来评测 example workspace
  - 测试用例示例：test/rl/test_example_workspace.py

## 构建 benchmark & example workspace
rdagent/scenarios/rl/eval/AutoRL-Bench/example_workspace

基于 Dockerfile `rdagent/scenarios/rl/eval/AutoRL-Bench/env/`
在 rdagent/utils/env.py 中编写 RL 专用的 DockerEnv


我们开始一个新任务。

## 构建一个 Agent 来生成 solution（workspace）

### 第一步：

参考 finetune agent，主要工作流在 `rdagent/scenarios/finetune/loop.py`

请按照上述结构为 RL-post-training 实现一个脚手架（scaffold）。


### 第二步：

在脚手架中实现具体的示例。


让我们从第一步开始；

请为 RL 场景添加一个入口，类似 rdagent/app/finetune/llm/loop.py


代码结构：
- `rdagent/scenarios/rl/`: 场景的具体功能实现
- `/rdagent/app/rl`: CLI 入口 & 配置

## 组件说明

CoSTEER：代码生成是困难的；我们需要多个步骤来生成代码。负责执行计划（计划来自外层循环）。
- for step in all_steps:
  - run step:
    - 当 step 是 coding 时，我们有内层循环来生成代码。


- TODO:
  - 简化脚手架逻辑
```


[[test/rl/test_example_workspace.py:6]]


## Coding Principles
Don't catch unknown exceptions when implementing new code. I prefer to let the error propagate so it can be detected and fixed promptly.

## (R)un 运行特定功能


```
```

### 调试用

## (A)I 编辑
 <发送给 AI 的指令>

## (E)xplanation 解释
 <理解代码的关键部分>

## (Q)uestions 问题
 <记录要问同事的问题>

 ## (B)acklogs 待办
 <设计改进>
