TODO: Add badges.

# News
| 🗞️News        | 📝Description                 |
| --            | ------                        |
| First release | RDAgent are release on Github |


# Introduction

RDAgent aims to automate the most critical and valuable aspects of the industrial R&D process, and we begins with focusing on the data-driven scenarios to streamline the development of models and data. 
Methodologically, we have identified a framework with two key components: 'R' for proposing new ideas and 'D' for implementing them.
We believe that the automatic evolution of R&D will lead to solutions of significant industrial value.


<!-- Tag Cloud -->
R&D is a very general scenario. The advent of RDAgent can be your
- [Automatic Quant Factory]()
- Data mining copilot: iteratively proposing data & models and implementing them by gaining knowledge from data.
- Research copilot: Auto read papers and implement model structures.
- ...

More methods and scenarios are being added to the project to empower your R&D processes and boost productivity.

We have a quick demo for one use case of RDAgent.
- TODO: Demo


# Quick start
You can try our demo by running the following command:

```bash
# prepare environment
# installation
# App entrance
```

# Scenarios


## Data-driven R&D
TODO: importance justification

### 🎯 Goal

In this project, we are aiming to build a Data-Centric R\&D Agent that can

+ Read real-world material (reports, papers, etc.) and extract key formulas, descriptions of interested features, factors and models.

+ Implement the extracted formulas, features, factors and models in runnable codes.
   + Due the limited ability for LLM in implementing in once, evolving the agent to be able to extend abilities by learn from feedback and knowledge and improve the agent's ability to implement more complex models.

+ Further propose new ideas based on current knowledge and observations.

![Data-Centric R&D Overview](docs/_static/overview.png)

### 🛣️ Brief Roadmap
In this section, we will briefly introduce the roadmap/technical type of this project.

1. Backbone LLM: We use GPT series as main backbone of the agent. `.env` file uis used to config settings (such as APIkey, APIEndpoint and etc) in the environment variables way. Check this [Readme](src/scripts/benchmark/README.md) for environment set up.

2. KnowledgeGraph based evolving: We do not do any further pertain or fine-tune on the LLM model. Instead, we modify prompts like RAG, but use knowledge graph query information to evolve the agent's ability to implement more complex models. 
   + Typically, we build a knowledge consisted with `Error`, `Component`(you can think of it as a numeric operation or function), `Trail` and etc. We add nodes of these types to the knowledge graph with relationship while the agent tries to implement a model. For each attempts, the agent will query the knowledge graph to get the information of current status as prompt input. The agent will also update the knowledge graph with the new information after the attempt.


## Code Refinement
Example: code standard, design. Lint


# 🔧 Development
- Set up the development environment.

   ```bash
   make dev
   ```

- Run linting and formatting.

   ```bash
   make lint
   ```

### 📚 Code Structure
1. Backbone/APIBackend of LLm are encapsulated in [src/finco/llm.py](src/finco/llm.py). All chat completion request are managed by this file.

2. All frequently modified codes under tense development are included in the [src/scripts](src/scripts) folder.
   + The most important task is to improve the agent's performance in the benchmark of factor implementation.

3. Currently, factor implementation is the main task. We define basic class of factor implementation in [src/scripts/factor_implementation/share_modules] and implementation strategies in [src/scripts/factor_implementation/baselines].


### 🔮 Future Code Structure

Currently, the code structure is unstable and will frequently change for quick updates. The code will be refactored before a standard release. Please try to align with the following principles when developing to minimize the effort required for future refactoring.

```
📂 src
➥ 📂 <project name>: avoid namespace
  ➥ 📁 core
  ➥ 📁 component A
  ➥ 📁 component B
  ➥ 📁 component C
  ➥ 📂 app
    ➥ 📁 scenario1
    ➥ 📁 scenario2
➥ 📁 scripts
```

| Folder Name    | Description                                                                                                                                                 |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 📁 core        | The core framework of the system. All classes should be abstract and usually can't be used directly.                                                        |
| 📁 component X | Useful components that can be used by others(e.g. scenario). Many subclasses of core classes are located here.                                                   |
| 📁 app         | Applications for specific scenarios (usually built based on components). Removing any of them does not affect the system's completeness or other scenarios. |
| 📁 scripts     | Quick and dirty things. These are candidates for core, components, and apps.                                                                                |



# Configuration:

You can manually source the `.env` file in your shell before running the Python script:
Most of the workflow are controlled by the environment variables.
```sh
# Export each variable in the .env file; Please note that it is different from `source .env` without export
export $(grep -v '^#' .env | xargs)
# Run the Python script
python your_script.py
```

# Naming convention

## File naming convention

| Name      | Description       |
| --        | --                |
| `conf.py` | The configuration for the module & app & project  | 

<!-- TODO: renaming files -->
 

# Contributing

## Guidance
This project welcomes contributions and suggestions.
You can find issues in the issues list or simply running `grep -r "TODO:"`.

Making contributions is not a hard thing. Solving an issue(maybe just answering a question raised in issues list ), fixing/issuing a bug, improving the documents and even fixing a typo are important contributions to RDAgent.


## Policy

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
