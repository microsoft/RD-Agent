# RD-Agent Copilot Instructions

RD-Agent is a Microsoft Research R&D automation framework that automates the **Research (propose hypotheses) → Development (implement)** cycle for data-driven ML tasks. Primary scenarios: quantitative finance (factor/model development via Qlib), data science/Kaggle competitions, and LLM fine-tuning.

## Build, Test & Lint

### Setup
```bash
make dev          # install all optional deps + pre-commit hook
make install      # editable install only
```

### Testing
```bash
make test                        # run all tests with coverage
make test-offline                # offline tests only (no LLM API calls)
pytest test/path/to/test_file.py::TestClass::test_method   # single test
pytest -m offline                # run only @pytest.mark.offline tests
```

Mark tests that don't call external APIs with `@pytest.mark.offline`. The `workspace/` directory is excluded from test discovery. Coverage threshold is 80%.

### Linting
```bash
make lint         # mypy + ruff + isort + black + toml-sort (check only)
make auto-lint    # auto-fix isort + black + toml-sort
make mypy         # type check rdagent/core (scope is expanding)
make ruff         # lint rdagent/core
```

- **Line length:** 120 (black + ruff)
- **isort profile:** black
- **mypy:** strict (`disallow_untyped_defs=true`, `warn_return_any=true`) — currently scoped to `rdagent/core/`
- Conventional commit prefixes required on PRs: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, etc.

## Architecture

### Package Layout

```
rdagent/
├── core/          # Abstract base classes and framework interfaces
├── components/    # Reusable building blocks (coder, runner, proposal, workflow, …)
├── scenarios/     # Domain-specific implementations (qlib, kaggle, data_science, …)
├── app/           # Entry points per scenario (qlib_rd_loop, data_science, kaggle, …)
├── oai/           # LLM backend abstraction (LiteLLM default, OpenAI, Azure)
├── log/           # Logging (loguru-based), Streamlit trace viewer UI
└── utils/         # Env execution, workflow loop, repo/blob utilities
```

### The R&D Loop

Each scenario runs a `LoopBase` (via `LoopMeta`) that orchestrates these steps:

```
HypothesisGen → Hypothesis2Experiment → Developer (coder) → Developer (runner) → Experiment2Feedback
     ↑                                                                                    |
     └────────────────────── Trace (accumulates history) ───────────────────────────────┘
```

`RDLoop` in `rdagent/components/workflow/rd_loop.py` is the canonical implementation. Each component is dynamically loaded by class path from a `BasePropSetting` config object.

### Core Abstractions (`rdagent/core/`)

| Class | Role |
|---|---|
| `Scenario` | Domain context — provides background, runtime environment description |
| `Developer[Exp]` | Transforms an experiment **in-place** (coder or runner) |
| `Evaluator` / `IterEvaluator` | Produces `Feedback`; iter variant uses coroutine `yield`/`send` |
| `EvolvingStrategy[T]` | Algorithm for one evolution iteration; yields partial states |
| `RAGStrategy[T]` | Manages a `EvolvingKnowledgeBase`: query, generate, dump, load |
| `Hypothesis` | Research idea with `hypothesis`, `reason`, `concise_*` fields |
| `ExperimentFeedback` | Boolean `decision` + `reason`; `bool(fb)` is `fb.decision` |
| `Workspace[Task, FB]` | Mutable container for code/data during task execution |
| `EvoStep[T]` | Dataclass: `evolvable_subjects`, `queried_knowledge`, `feedback` |

**Critical Developer contract:** `develop(exp)` mutates `exp` in-place. Do not return a new object.

### Configuration

All settings use **Pydantic v2 `BaseSettings`** via `ExtendedBaseSettings` (supports env-prefix inheritance across base classes):

```python
from rdagent.core.conf import ExtendedBaseSettings, RD_AGENT_SETTINGS
from rdagent.oai.llm_conf import LLM_SETTINGS
```

Settings are overridable via environment variables or a `.env` file in the working directory (auto-loaded by the CLI). Key settings:
- `RD_AGENT_SETTINGS.workspace_path` — where experiment artifacts are written (`git_ignore_folder/` by default)
- `LLM_SETTINGS.backend` — LLM provider class path (default: `rdagent.oai.backend.LiteLLMAPIBackend`)
- `LLM_SETTINGS.chat_model` / `embedding_model`

### LLM Access

```python
from rdagent.oai.llm_utils import APIBackend
response = APIBackend().build_messages_and_create_chat_completion(...)
embeddings = APIBackend().create_embedding(str_list)
```

`APIBackend` is a factory that returns the configured backend. LLM calls are cached via MD5-keyed pickle when `LLM_SETTINGS.use_chat_cache = True`.

### Prompts

Prompts live in YAML files alongside the module that uses them. They are loaded via `Prompts(file_path)` (a `SingletonBaseClass` dict). Jinja2-style templating is used for variable substitution. Prompt files are typically at `rdagent/{component}/prompts.yaml`.

### Logging

```python
from rdagent.log import rdagent_logger as logger
logger.info("message")
logger.log_object(obj, tag="label")   # structured object logging
```

`rdagent_logger` is a global `RDAgentLog` singleton backed by loguru. Use `logger.log_object()` for structured data (scenarios, settings, experiments). The Streamlit UI (`rdagent ui`) reads these logs for visualization.

## Key Conventions

- **`from __future__ import annotations`** at the top of every module — required for forward references with mypy.
- **TypeVars are bound**: `ASpecificExp = TypeVar("ASpecificExp", bound=Experiment)`. Follow this pattern when adding generics.
- **`SingletonBaseClass`**: kwargs-only construction enforced. Do not pass positional args to singletons.
- **`import_class(dotted.path)`** from `rdagent.core.utils` — used everywhere to dynamically load scenario components from config strings. Use this rather than direct imports when the class path is user-configurable.
- **`git_ignore_folder/`** — runtime workspace output; already gitignored. Experiment artifacts, pickle caches, and checkpoints go here.
- **Parallel loops**: `RD_AGENT_SETTINGS.step_semaphore` controls concurrency. When `> 1`, subprocesses are used automatically.
- **Ruff ignore list**: `ANN401`, `D` (docstrings), `ERA001`, `T20` (print), `S101` (assert), `TD`/`FIX` (todos) are intentionally suppressed project-wide.
