# 开发计划：并行多 Trace 探索 (最终版)

## 1. 核心设计思想

采用 **"显式上下文传递与属性注入 (Explicit Context Passing & Attribute Injection)"** 方案。

**目标**：在最小化改动核心循环 (`loop.py`) 的前提下，实现可由配置开关控制的、并发安全的多 trace 并行探索功能。

**核心创新**：通过显式的 `local_selection` 参数传递，完全消除对共享状态 `trace.current_selection` 的依赖，从根本上避免了 Race Condition。

## 1.1. 技术方案对比

### 原始问题分析
在并行环境中，传统的共享状态方式存在严重的 Race Condition：
```python
# 问题场景：
# 任务A: trace.set_current_selection(parent_A)
# 任务B: trace.set_current_selection(parent_B)  # 覆盖了任务A的设置！
# 任务A: exp = exp_gen.gen(trace)  # 读取到的是parent_B，而不是parent_A
```

### 解决方案：显式上下文传递
```python
# 新方案：
# 任务A: local_selection_A = scheduler.select_trace()
# 任务A: exp_A = exp_gen.gen(trace, local_selection_A)  # 显式传递
# 任务B: local_selection_B = scheduler.select_trace() 
# 任务B: exp_B = exp_gen.gen(trace, local_selection_B)  # 显式传递
```

**优势**：
- ✅ **完全消除 Race Condition**：每个任务使用自己的局部变量
- ✅ **向后兼容**：`local_selection=None` 时使用传统模式
- ✅ **简洁高效**：无需复杂的锁机制
- ✅ **易于理解**：上下文传递路径清晰可见

## 2. 各模块代码修改逻辑梳理

### 2.1. `DSExperiment` (上下文载体)
-   **文件**: `rdagent/scenarios/data_science/experiment/experiment.py`
-   **做什么**: 为 `DSExperiment` 类增加：
    - 属性：`self.local_selection: tuple[int, ...] | None = None`
    - 方法：`set_local_selection(local_selection: tuple[int, ...]) -> None`
-   **为什么**: 这是实现"属性注入"的载体。在并行模式下，实验对象需要"记住"自己应该连接到哪个父节点。这个属性就像是给实验贴上了一个"寻根"标签，确保在后续的 `record` 阶段能够正确地将实验连接到 trace 图的正确位置。

### 2.2. `TraceScheduler` (决策者)
-   **文件**: `rdagent/scenarios/data_science/proposal/exp_gen/trace_scheduler.py`
-   **做什么**: 创建调度器模块，包含：
    - `TraceScheduler` 抽象基类：定义调度接口
    - `RoundRobinScheduler` 实现类：轮询调度策略
-   **为什么**: 当存在多个可供探索的 trace 分支时，需要一个明确的策略来决定"下一个扩展哪一个？"。调度器内部使用 `asyncio.Lock` 保护自身状态，确保在并发环境中能够安全、公平地分配探索目标。

### 2.3. `ParallelMultiTraceExpGen` (总控制器)
-   **文件**: `rdagent/scenarios/data_science/proposal/exp_gen/parallel.py`
-   **做什么**: 创建并行模式的总控制器，核心逻辑：
    ```python
    async def async_gen(self, trace: DSTrace, loop: LoopBase) -> DSExperiment:
        # 步骤1：智能选择探索目标
        if trace.sub_trace_count < self.target_trace_count:
            local_selection = trace.NEW_ROOT  # 创建新分支
        else:
            local_selection = await self.trace_scheduler.select_trace(trace)  # 扩展现有分支
        
        # 步骤2：等待执行槽位
        while True:
            if loop.get_unfinished_loop_cnt(loop.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                # 步骤3：显式传递上下文，生成实验
                exp = self.exp_gen.gen(trace, local_selection)
                
                # 步骤4：属性注入，携带上下文
                exp.set_local_selection(local_selection)
                
                return exp
            await asyncio.sleep(1)
    ```
-   **为什么**: 这是并行模式的核心协调者，负责：
    1. **智能决策**：根据当前分支数量决定是创建新分支还是扩展现有分支
    2. **显式传递**：将选定的上下文作为参数传递给底层生成器
    3. **属性注入**：确保实验对象携带正确的父节点信息
    4. **并发控制**：等待合适的执行时机

### 2.4. `DSProposalV2ExpGen` (底层生成器)
-   **文件**: `rdagent/scenarios/data_science/proposal/exp_gen/proposal.py`
-   **做什么**: 修改 `gen()` 方法签名和实现：
    ```python
    def gen(self, trace: DSTrace, local_selection: tuple[int, ...] | None = None) -> DSExperiment:
        # 使用显式的 local_selection 而不是 trace.current_selection
        if local_selection is None:
            # 传统模式：使用全局状态
            exp_feedback_list_desc = trace.experiment_and_feedback_list_after_init(return_type="all")
        else:
            # 并行模式：使用局部上下文
            exp_feedback_list_desc = trace.experiment_and_feedback_list_after_init(
                return_type="all", 
                search_type="ancestors",
                selection=local_selection
            )
    ```
-   **为什么**: 这是消除 Race Condition 的关键。通过支持可选的 `local_selection` 参数：
    - **并行安全**：每个任务使用自己的局部上下文，互不干扰
    - **向后兼容**：当 `local_selection=None` 时，保持原有行为
    - **上下文精确**：只获取特定分支的相关信息，避免噪音

### 2.5. `DataScienceRDLoop` (核心循环)
-   **文件**: `rdagent/scenarios/data_science/loop.py`
-   **做什么**:
    1. **简化 `direct_exp_gen`**：移除检查点选择逻辑，专注于调用 `exp_gen.async_gen()`
    2. **关键修改：将 `record` 方法从同步改为异步**：
       - 原来：`def record(self, prev_out: dict[str, Any]) -> dict[str, Any]`
       - 现在：`async def record(self, prev_out: dict[str, Any]) -> dict[str, Any]`
    3. **增强 `_perform_record`**：添加状态同步逻辑：
    ```python
    def _perform_record(self, prev_out: dict[str, Any]):
        # 获取实验对象
        exp = prev_out.get("running") or prev_out.get("direct_exp_gen") or prev_out.get("coding")
        
        # 关键：使用实验携带的局部上下文更新全局状态
        if hasattr(exp, "local_selection") and exp.local_selection is not None:
            self.trace.set_current_selection(exp.local_selection)
        
        # 同步 DAG 结构并记录实验
        self.trace.sync_dag_parent_and_hist()
        self.trace.hist.append((exp, feedback))
    ```
-   **为什么**: 这是整个方案的收尾环节：
    - **异步兼容性**：将 `record` 改为 `async` 确保与整个异步工作流的兼容性，为未来可能的异步扩展做准备
    - **状态同步**：将实验携带的局部上下文同步回全局状态
    - **结构维护**：确保 trace 图的 DAG 结构正确
    - **最小侵入**：对核心循环的修改极其有限

## 3. 工作流程详解

### 3.1. 并行生成阶段
```
任务A时间线：
T1: local_selection_A = scheduler.select_trace()    # 获得 (leaf_5,)
T2: exp_A = exp_gen.gen(trace, local_selection_A)   # 使用局部上下文
T3: exp_A.set_local_selection(local_selection_A)    # 注入属性
T4: return exp_A                                    # 携带标签的实验

任务B时间线（并行进行）：
T1: local_selection_B = scheduler.select_trace()    # 获得 (leaf_7,)
T2: exp_B = exp_gen.gen(trace, local_selection_B)   # 使用局部上下文
T3: exp_B.set_local_selection(local_selection_B)    # 注入属性
T4: return exp_B                                    # 携带标签的实验
```

### 3.2. 记录同步阶段
```
记录阶段（串行进行）：
1. 实验A完成 -> _perform_record(exp_A)
   - trace.set_current_selection(exp_A.local_selection)  # 恢复上下文
   - trace.hist.append((exp_A, feedback_A))             # 记录到正确位置

2. 实验B完成 -> _perform_record(exp_B)  
   - trace.set_current_selection(exp_B.local_selection)  # 恢复上下文
   - trace.hist.append((exp_B, feedback_B))             # 记录到正确位置
```

## 4. 关键技术特性

### 4.1. 并发安全性
- **无共享状态竞争**：每个任务使用独立的 `local_selection` 变量
- **调度器内部保护**：`RoundRobinScheduler` 使用 `asyncio.Lock` 保护自身状态
- **记录阶段串行**：`_perform_record` 天然串行执行，无需额外锁

### 4.2. 智能分支管理
- **动态分支创建**：当分支数 < 目标数时自动创建新分支
- **公平轮询调度**：使用 Round-Robin 策略确保所有分支得到公平探索
- **自适应调整**：调度器能自动适应 trace 图的动态变化

### 4.3. 向后兼容性
- **渐进式启用**：通过配置开关控制，不影响现有功能
- **参数可选**：`local_selection` 参数默认为 `None`，保持原有行为
- **接口一致**：对外接口保持不变，内部实现透明升级

## 5. 使用方式

### 5.1. 配置启用
```python
# 在配置中启用并行多trace模式
DS_RD_SETTING.enable_parallel_multi_trace = True
DS_RD_SETTING.max_traces = 3  # 最大并行分支数
```

### 5.2. ExpGen选择
```python
# 使用并行生成器
exp_gen = ParallelMultiTraceExpGen(scen)

# 或在配置中指定
PROP_SETTING.hypothesis_gen = "rdagent.scenarios.data_science.proposal.exp_gen.ParallelMultiTraceExpGen"
```

### 5.3. 调度策略配置
```python
# 可以扩展不同的调度策略
class PriorityScheduler(TraceScheduler):
    async def select_trace(self, trace: DSTrace) -> tuple[int, ...]:
        # 基于优先级的调度逻辑
        pass
```

---

# Development Plan: Parallel Multi-Trace Exploration (Final Version)

## 1. Core Design Philosophy

Adopt the **"Explicit Context Passing & Attribute Injection"** approach.

**Goal**: To implement a concurrently safe, multi-trace parallel exploration feature that can be toggled by configuration, all while minimizing modifications to the core loop (`loop.py`).

**Core Innovation**: By using explicit `local_selection` parameter passing, we completely eliminate dependency on the shared state `trace.current_selection`, fundamentally avoiding Race Conditions.

## 1.1. Technical Approach Comparison

### Original Problem Analysis
In parallel environments, the traditional shared state approach has serious Race Conditions:
```python
# Problem scenario:
# Task A: trace.set_current_selection(parent_A)
# Task B: trace.set_current_selection(parent_B)  # Overwrites Task A's setting!
# Task A: exp = exp_gen.gen(trace)  # Reads parent_B instead of parent_A
```

### Solution: Explicit Context Passing
```python
# New approach:
# Task A: local_selection_A = scheduler.select_trace()
# Task A: exp_A = exp_gen.gen(trace, local_selection_A)  # Explicit passing
# Task B: local_selection_B = scheduler.select_trace() 
# Task B: exp_B = exp_gen.gen(trace, local_selection_B)  # Explicit passing
```

**Advantages**:
- ✅ **Complete Race Condition Elimination**: Each task uses its own local variables
- ✅ **Backward Compatible**: Traditional mode when `local_selection=None`
- ✅ **Simple and Efficient**: No complex locking mechanisms needed
- ✅ **Easy to Understand**: Clear and visible context passing path

## 2. Code Modification Logic by Module

### 2.1. `DSExperiment` (Context Carrier)
-   **File**: `rdagent/scenarios/data_science/experiment/experiment.py`
-   **Action**: Add to the `DSExperiment` class:
    - Attribute: `self.local_selection: tuple[int, ...] | None = None`
    - Method: `set_local_selection(local_selection: tuple[int, ...]) -> None`
-   **Rationale**: This is the carrier for "attribute injection". In parallel mode, experiment objects need to "remember" which parent node they should connect to. This attribute acts as a "return address" label for the experiment, ensuring correct connection to the trace graph during the subsequent `record` phase.

### 2.2. `TraceScheduler` (Decision Maker)
-   **File**: `rdagent/scenarios/data_science/proposal/exp_gen/trace_scheduler.py`
-   **Action**: Create scheduler module containing:
    - `TraceScheduler` abstract base class: Define scheduling interface
    - `RoundRobinScheduler` implementation: Round-robin scheduling strategy
-   **Rationale**: When multiple trace branches are available for exploration, we need a clear strategy to decide "which one to expand next?". The scheduler uses internal `asyncio.Lock` to protect its own state, ensuring safe and fair target allocation in concurrent environments.

### 2.3. `ParallelMultiTraceExpGen` (Main Controller)
-   **File**: `rdagent/scenarios/data_science/proposal/exp_gen/parallel.py`
-   **Action**: Create the main controller for parallel mode with core logic:
    ```python
    async def async_gen(self, trace: DSTrace, loop: LoopBase) -> DSExperiment:
        # Step 1: Intelligent target selection
        if trace.sub_trace_count < self.target_trace_count:
            local_selection = trace.NEW_ROOT  # Create new branch
        else:
            local_selection = await self.trace_scheduler.select_trace(trace)  # Extend existing branch
        
        # Step 2: Wait for execution slot
        while True:
            if loop.get_unfinished_loop_cnt(loop.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                # Step 3: Explicit context passing, generate experiment
                exp = self.exp_gen.gen(trace, local_selection)
                
                # Step 4: Attribute injection, carry context
                exp.set_local_selection(local_selection)
                
                return exp
            await asyncio.sleep(1)
    ```
-   **Rationale**: This is the core coordinator for parallel mode, responsible for:
    1. **Intelligent Decision**: Decide whether to create new branches or extend existing ones based on current branch count
    2. **Explicit Passing**: Pass selected context as parameter to underlying generator
    3. **Attribute Injection**: Ensure experiment objects carry correct parent node information
    4. **Concurrency Control**: Wait for appropriate execution timing

### 2.4. `DSProposalV2ExpGen` (Underlying Generator)
-   **File**: `rdagent/scenarios/data_science/proposal/exp_gen/proposal.py`
-   **Action**: Modify `gen()` method signature and implementation:
    ```python
    def gen(self, trace: DSTrace, local_selection: tuple[int, ...] | None = None) -> DSExperiment:
        # Use explicit local_selection instead of trace.current_selection
        if local_selection is None:
            # Traditional mode: use global state
            exp_feedback_list_desc = trace.experiment_and_feedback_list_after_init(return_type="all")
        else:
            # Parallel mode: use local context
            exp_feedback_list_desc = trace.experiment_and_feedback_list_after_init(
                return_type="all", 
                search_type="ancestors",
                selection=local_selection
            )
    ```
-   **Rationale**: This is key to eliminating Race Conditions. By supporting optional `local_selection` parameter:
    - **Parallel Safe**: Each task uses its own local context without interference
    - **Backward Compatible**: Maintains original behavior when `local_selection=None`
    - **Context Precise**: Only retrieves information relevant to specific branches, avoiding noise

### 2.5. `DataScienceRDLoop` (Core Loop)
-   **File**: `rdagent/scenarios/data_science/loop.py`
-   **Action**:
    1. **Simplify `direct_exp_gen`**: Remove checkpoint selection logic, focus on calling `exp_gen.async_gen()`
    2. **Key Modification: Convert `record` method from sync to async**:
       - Before: `def record(self, prev_out: dict[str, Any]) -> dict[str, Any]`
       - After: `async def record(self, prev_out: dict[str, Any]) -> dict[str, Any]`
    3. **Enhance `_perform_record`**: Add state synchronization logic:
    ```python
    def _perform_record(self, prev_out: dict[str, Any]):
        # Get experiment object
        exp = prev_out.get("running") or prev_out.get("direct_exp_gen") or prev_out.get("coding")
        
        # Key: Use experiment's local context to update global state
        if hasattr(exp, "local_selection") and exp.local_selection is not None:
            self.trace.set_current_selection(exp.local_selection)
        
        # Synchronize DAG structure and record experiment
        self.trace.sync_dag_parent_and_hist()
        self.trace.hist.append((exp, feedback))
    ```
-   **Rationale**: This is the final step of the entire approach:
    - **Async Compatibility**: Converting `record` to `async` ensures compatibility with the entire async workflow and prepares for potential future async extensions
    - **State Synchronization**: Sync experiment's local context back to global state
    - **Structure Maintenance**: Ensure correct DAG structure of trace graph
    - **Minimal Intrusion**: Extremely limited modifications to core loop

## 3. Workflow Details

### 3.1. Parallel Generation Phase
```
Task A Timeline:
T1: local_selection_A = scheduler.select_trace()    # Gets (leaf_5,)
T2: exp_A = exp_gen.gen(trace, local_selection_A)   # Uses local context
T3: exp_A.set_local_selection(local_selection_A)    # Inject attribute
T4: return exp_A                                    # Experiment with label

Task B Timeline (parallel):
T1: local_selection_B = scheduler.select_trace()    # Gets (leaf_7,)
T2: exp_B = exp_gen.gen(trace, local_selection_B)   # Uses local context
T3: exp_B.set_local_selection(local_selection_B)    # Inject attribute
T4: return exp_B                                    # Experiment with label
```

### 3.2. Recording Synchronization Phase
```
Recording Phase (serial):
1. Experiment A completes -> _perform_record(exp_A)
   - trace.set_current_selection(exp_A.local_selection)  # Restore context
   - trace.hist.append((exp_A, feedback_A))             # Record to correct position

2. Experiment B completes -> _perform_record(exp_B)  
   - trace.set_current_selection(exp_B.local_selection)  # Restore context
   - trace.hist.append((exp_B, feedback_B))             # Record to correct position
```

## 4. Key Technical Features

### 4.1. Concurrency Safety
- **No Shared State Competition**: Each task uses independent `local_selection` variables
- **Internal Scheduler Protection**: `RoundRobinScheduler` uses `asyncio.Lock` to protect its own state
- **Serial Recording Phase**: `_perform_record` executes serially by nature, no additional locks needed

### 4.2. Intelligent Branch Management
- **Dynamic Branch Creation**: Automatically creates new branches when branch count < target
- **Fair Round-Robin Scheduling**: Uses Round-Robin strategy to ensure fair exploration of all branches
- **Adaptive Adjustment**: Scheduler automatically adapts to dynamic changes in trace graph

### 4.3. Backward Compatibility
- **Progressive Enablement**: Controlled by configuration switches without affecting existing functionality
- **Optional Parameters**: `local_selection` parameter defaults to `None`, maintaining original behavior
- **Consistent Interface**: External interface remains unchanged, internal implementation transparently upgraded

## 5. Usage

### 5.1. Configuration Enablement
```python
# Enable parallel multi-trace mode in configuration
DS_RD_SETTING.enable_parallel_multi_trace = True
DS_RD_SETTING.max_traces = 3  # Maximum parallel branches
```

### 5.2. ExpGen Selection
```python
# Use parallel generator
exp_gen = ParallelMultiTraceExpGen(scen)

# Or specify in configuration
PROP_SETTING.hypothesis_gen = "rdagent.scenarios.data_science.proposal.exp_gen.ParallelMultiTraceExpGen"
```

### 5.3. Scheduling Strategy Configuration
```python
# Can extend different scheduling strategies
class PriorityScheduler(TraceScheduler):
    async def select_trace(self, trace: DSTrace) -> tuple[int, ...]:
        # Priority-based scheduling logic
        pass
```
