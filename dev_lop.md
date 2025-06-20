# 开发计划：并行多 Trace 探索 (最终版)

## 1. 核心设计思想

采用 **"属性注入与轻量级锁定 (Attribute Injection & Minimal Locking)"** 方案。

**目标**：在最小化改动核心循环 (`loop.py`) 的前提下，实现可由配置开关控制的、并发安全的多 trace 并行探索功能。

## 2. 各模块代码修改逻辑梳理

### 2.1. `DSExperiment` (上下文载体)
-   **文件**: `rdagent/scenarios/data_science/experiment/experiment.py`
-   **做什么**: 为 `DSExperiment` 类增加一个新属性 `self.parent_selection: tuple[int, ...] | None = None`。
-   **为什么**: 这是实现"属性注入"的核心。在多分支并行模式下，当一个实验被创建时，我们必须确定它未来要连接到哪个父节点上。这个属性就像是给实验贴上了一个"寻根"的标签，解决了并发场景下"上下文丢失"的问题。后续流程无需关心此标签，但最终的 `record` 步骤需要靠它来准确地将实验连接到 `trace` 图的正确位置。

### 2.2. `TraceScheduler` (决策者)
-   **文件**: `rdagent/scenarios/data_science/proposal/exp_gen/scheduler.py` (新建)
-   **做什么**: 创建一个全新的调度器模块，包含 `TraceScheduler` 抽象基类和 `RoundRobinScheduler` 实现类。
-   **为什么**: 当存在多个可供探索的 trace 分支时，我们需要一个明确的策略来决定"下一个扩展哪一个？"。`TraceScheduler` 将这个决策逻辑独立封装，其内部的 `asyncio.Lock` 确保了即使多个并行的生成任务同时请求决策，调度器也能安全、无冲突地分配目标。

### 2.3. `ParallelExpGen` (总控制器)
-   **文件**: `rdagent/scenarios/data_science/proposal/exp_gen/parallel.py` (新建)
-   **做什么**: 创建一个全新的 `ParallelExpGen` 类，作为并行模式的总控制器和"属性注入"的执行者。
-   **为什么**: 当并行模式开启时，此类被调用。它的职责是：
    1.  调用 `TraceScheduler` 来获取一个父节点 (`parent_selection`)。
    2.  调用底层的标准实验生成器来创建一个 `DSExperiment` 实例。
    3.  执行最关键的一步：将获取到的 `parent_selection` **注入**到这个新的实验实例中。
    4.  返回这个携带了"父节点"标签的实验对象。

### 2.4. `RDLoop` (基础框架)
-   **文件**: `rdagent/utils/workflow/loop.py`
-   **做什么**: 将 `asyncio.iscoroutinefunction` 替换为 `inspect.iscoroutinefunction`。
-   **为什么**: 这是一个为了代码更健壮、更符合 Python 标准库规范的微小改进，确保了框架未来能更好地处理各种异步函数，使其更具通用性。

### 2.5. `DataScienceRDLoop` (核心循环)
-   **文件**: `rdagent/scenarios/data_science/loop.py`
-   **做什么**:
    1.  在 `__init__` 中，根据配置增加一个可开关的 `asyncio.Lock` 实例。
    2.  简化 `direct_exp_gen` 的逻辑，将其不再负责选择检查点，这个职责被正确地移交给了 `ExpGen` 的子类。
    3.  将 `record` 方法从 `def` 改为 `async def`。
    4.  在 `record` 方法内部，使用创建的锁来包裹核心的 `trace` 修改操作。
-   **为什么**: 这是保证并发安全的最后一环，也是对核心循环影响最小的方案。`record` 是唯一会"写入"共享数据 `trace` 的地方。在并行模式下，通过在这个极小的、执行速度极快的代码块前后加锁和释放锁，我们确保了即使有多个实验同时完成，它们也会严格排队、一个接一个地被记录到 `trace` 中，从而完美地避免了数据图的损坏。当并行模式关闭时，这个锁不会被创建，`record` 的行为也与原来完全一致。

---

# Development Plan: Parallel Multi-Trace Exploration (Final Version)

## 1. Core Design Philosophy

Adopt the **"Attribute Injection & Minimal Locking"** approach.

**Goal**: To implement a concurrently safe, multi-trace parallel exploration feature that can be toggled by configuration, all while minimizing modifications to the core loop (`loop.py`).

## 2. Code Modification Logic by Module

### 2.1. `DSExperiment` (The Context Carrier)
-   **File**: `rdagent/scenarios/data_science/experiment/experiment.py`
-   **Action**: Add a new attribute `self.parent_selection: tuple[int, ...] | None = None` to the `DSExperiment` class.
-   **Rationale**: This is the core of the "attribute injection" strategy. In a multi-branch parallel mode, we must determine an experiment's future parent node at the moment of its creation. This attribute acts as a "return address" label for the experiment, solving the "lost context" problem in concurrent scenarios. Subsequent stages (`coding`, `running`) are agnostic to this label, but the final `record` step relies on it to accurately connect the experiment to the correct place in the `trace` graph.

### 2.2. `TraceScheduler` (The Decision-Maker)
-   **File**: `rdagent/scenarios/data_science/proposal/exp_gen/scheduler.py` (New File)
-   **Action**: Create a new scheduler module containing the `TraceScheduler` ABC and a `RoundRobinScheduler` implementation.
-   **Rationale**: When multiple trace branches (i.e., leaves of the DAG) are available for exploration, we need a clear strategy to decide "which one to expand next?". The `TraceScheduler` encapsulates this decision logic, and its internal `asyncio.Lock` ensures that it can safely and correctly assign targets even when called by multiple concurrent generation tasks.

### 2.3. `ParallelExpGen` (The Controller)
-   **File**: `rdagent/scenarios/data_science/proposal/exp_gen/parallel.py` (New File)
-   **Action**: Create a new `ParallelExpGen` class to act as the controller for parallel mode and the executor of "attribute injection".
-   **Rationale**: When parallel mode is enabled, this class is invoked. Its responsibilities are:
    1.  Call the `TraceScheduler` to get a parent node (`parent_selection`).
    2.  Call the underlying, standard experiment generator to create a `DSExperiment` instance.
    3.  Perform the most critical step: **inject** the retrieved `parent_selection` into this new experiment instance.
    4.  Return the experiment object, which now carries its parent-node label.

### 2.4. `RDLoop` (The Base Framework)
-   **File**: `rdagent/utils/workflow/loop.py`
-   **Action**: Replace `asyncio.iscoroutinefunction` with `inspect.iscoroutinefunction`.
-   **Rationale**: This is a minor improvement for code robustness and compliance with Python standard library conventions, ensuring the framework is more versatile and can better handle various asynchronous functions in the future.

### 2.5. `DataScienceRDLoop` (The Core Loop)
-   **File**: `rdagent/scenarios/data_science/loop.py`
-   **Action**:
    1.  In `__init__`, add a toggleable `asyncio.Lock` instance based on configuration.
    2.  Simplify the logic in `direct_exp_gen`, delegating the responsibility of checkpoint selection to the `ExpGen` subclass where it belongs.
    3.  Change the `record` method signature from `def` to `async def`.
    4.  Inside the `record` method, wrap the core `trace` modification operations with the created lock.
-   **Rationale**: This is the final and most crucial step for ensuring concurrency safety, and it's the least invasive approach for the core loop. The `record` method is the only place that "writes" to the shared `trace` data. In parallel mode, by acquiring and releasing a lock around this tiny, fast block of code, we guarantee that even if multiple experiments finish simultaneously, they are recorded into the `trace` one by one, perfectly avoiding data graph corruption. When parallel mode is disabled, the lock is not created, and the `record` behavior remains identical to the original.
