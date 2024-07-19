TODO: Add badges.

# News
| 🗞️News        | 📝Description                 |
| --            | ------                        |
| First release | RDAgent are release on Github |


# Introduction

TODO: A piture to introduce the project scenario.

RDAgent aims to automate the most critical and valuable aspects of the industrial R&D process, and we begins with focusing on the data-driven scenarios to streamline the development of models and data. 
Methodologically, we have identified a framework with two key components: 'R' for proposing new ideas and 'D' for implementing them.
We believe that the automatic evolution of R&D will lead to solutions of significant industrial value.


<!-- Tag Cloud -->
R&D is a very general scenario. The advent of RDAgent can be your
- [🎥Automatic Quant Factory]()
- Data mining copilot: iteratively proposing [🎥data]() & [models]() and implementing them by gaining knowledge from data.
- Research copilot: Auto read [🎥research papers]()/[🎥reports]() and implement model structures or building datasets.
- ...

You can click the [🎥link]() above to view the demo. More methods and scenarios are being added to the project to empower your R&D processes and boost productivity.

We have a quick 🎥demo for one use case of RDAgent.
- TODO: Demo


# ⚡Quick start
You can try our demo by running the following command:

```bash
# TODO:
# prepare environment
# installation
# App entrance
```

The [🎥demo]() is implemented by the above commands.

# Scenarios

We have applied RD-Agent to multiple valuable data-driven industrial scenarios..


## 🎯 Goal: Agent for Data-driven R&D

In this project, we are aiming to build a Agent to automate Data-Driven R\&D that can
+ 📄Read real-world material (reports, papers, etc.) and **extract** key formulas, descriptions of interested **features** and **models**, which are the key components of data-driven R&D .
+ 🛠️**Implement** the extracted formulas, features, factors and models in runnable codes.
   + Due the limited ability for LLM in implementing in once, evolving the agent to be able to extend abilities by learn from feedback and knowledge and improve the agent's ability to implement more complex models.
+ 💡Propose **new ideas** based on current knowledge and observations.

![Data-Centric R&D Overview](docs/_static/overview.png)

## 📈 Scenarios Matrix 
Here is our supported scenarios

| Scenario/Target | Model Implementation                                                                     | Data Building                                                                            |
| --              | --                                                                                       | --                                                                                       |
| 💹Finance         | Auto paper/reports reading & implementation <br/> Iteratively Proposing Ideas & Evolving | Auto paper/reports reading & implementation <br/> Iteratively Proposing Ideas & Evolving |
| 🩺Medical         | Iteratively Proposing Ideas & Evolving                                                   | -                                                                                        |

Different scenarios vary in entrance and configuration. Please check the detailed setup tutorial in the scenarios documents.

# Framework

TODO: a picture of the framework.


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
