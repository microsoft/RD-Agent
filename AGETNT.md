# 多 Agent GitHub 协作约束

本文档定义多个 Agent 同时参与本项目开发时的 GitHub 工作流约束。目标是避免互相覆盖、降低合并冲突，并确保每次变更都有独立审查记录。

## 基本原则

- 所有代码修改必须通过 Pull Request 提交，不直接向 `main` 或其他保护分支推送。
- 每个 Agent 每次只处理一个明确任务，避免在同一个 PR 中混入无关修改。
- 每个 PR 必须由另一个 Agent 审查并合并，提交该 PR 的 Agent 不得自行合并。
- 修改前必须先获取最新远端代码，确认自己的工作基于当前最新状态。
- 不得回滚、覆盖或清理其他 Agent 的改动，除非该改动明确属于当前任务且已经在 PR 中说明。

## 开始修改前

1. 获取最新代码:

   ```bash
   git fetch origin
   ```

2. 确认本地工作区状态:

   ```bash
   git status --short --branch
   ```

3. 基于最新主分支创建任务分支:

   ```bash
   git switch main
   git pull --ff-only origin main
   git switch -c agent/<agent-name>/<short-task-name>
   ```

4. 如果本地存在未提交改动，先判断来源:
   - 自己上一个任务的改动: 提交、stash 或切换到对应分支继续处理。
   - 其他 Agent 的改动: 不修改、不删除、不格式化，必要时先沟通。
   - 无法确认来源的改动: 暂停并记录，避免误覆盖。

## 分支命名

推荐格式:

```text
agent/<agent-name>/<short-task-name>
```

示例:

```text
agent/codex/update-github-workflow-doc
agent/claude/fix-data-probe-retry
agent/gemini/add-factor-report-test
```

分支名应简短、可读，只描述当前任务。

## 提交要求

- 提交必须只包含当前任务相关文件。
- 提交前检查暂存内容:

  ```bash
  git diff --cached --stat
  git diff --cached
  ```

- 提交信息使用祈使句，说明变更目的:

  ```bash
  git commit -m "Document multi-agent PR workflow"
  ```

- 如果仓库已有格式化或测试命令，提交前运行与本次改动相关的最小验证。
- 文档类改动至少检查 Markdown 渲染结构、标题层级和命令示例是否正确。

## Pull Request 要求

PR 描述必须包含:

- 背景: 为什么需要这次修改。
- 变更: 修改了哪些文件和行为。
- 验证: 运行过哪些检查；如果未运行，说明原因。
- 风险: 是否可能影响其他 Agent 的工作、配置或数据。

PR 标题建议格式:

```text
[agent-name] Short task summary
```

示例:

```text
[codex] Document multi-agent PR workflow
```

## 审查与合并规则

- PR 作者不得合并自己的 PR。
- 一个 Agent 创建的 PR 必须由另一个 Agent 审查。
- 审查 Agent 至少检查:
  - 是否基于最新 `main`。
  - 是否只包含当前任务相关改动。
  - 是否误改、删除或格式化了其他 Agent 的文件。
  - 是否有必要的验证记录。
  - 是否存在明显的测试缺口、运行风险或配置风险。
- 合并前如果 `main` 有新提交，PR 作者需要 rebase 或 merge 最新代码，并重新验证。
- 合并方式优先使用 Squash and merge，保持主分支历史清晰。

## 冲突处理

- 遇到冲突时，由 PR 作者负责解决。
- 解决冲突前先确认冲突文件中哪些内容来自其他 Agent。
- 不理解的冲突不得靠删除对方代码解决，应在 PR 中说明并请求对应 Agent 复核。
- 冲突解决后重新运行相关验证，并在 PR 中更新说明。

## 禁止事项

- 禁止直接 force push 到共享分支。
- 禁止在未确认来源的情况下执行破坏性命令，例如:

  ```bash
  git reset --hard
  git checkout -- .
  git clean -fd
  ```

- 禁止把本地实验输出、缓存、密钥、`.env` 敏感配置提交到 PR。
- 禁止在同一个 PR 中混合代码、数据、格式化、依赖升级和无关重构。
- 禁止绕过另一个 Agent 的审查直接合并。

## 推荐日常流程

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c agent/<agent-name>/<short-task-name>

# 修改代码或文档

git status --short
git diff
git add <changed-files>
git diff --cached
git commit -m "<clear commit message>"
git push -u origin agent/<agent-name>/<short-task-name>
```

随后在 GitHub 创建 PR，并指定另一个 Agent 审查。审查通过后由审查 Agent 合并。

## 多 Agent 交接记录

当任务需要交接给另一个 Agent 时，在 PR 或项目文档中保留以下信息:

- 当前目标和未完成事项。
- 已修改文件列表。
- 已运行命令和验证结果。
- 已知风险、失败日志或外部依赖。
- 不应触碰的本地改动或数据文件。

交接记录应足够具体，使接手 Agent 不需要猜测当前状态。
