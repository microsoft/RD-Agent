[![CI](https://github.com/microsoft/RD-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/ci.yml)
[![CodeQL](https://github.com/microsoft/RD-Agent/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/github-code-scanning/codeql)
[![Dependabot Updates](https://github.com/microsoft/RD-Agent/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/dependabot/dependabot-updates)
[![Lint PR Title](https://github.com/microsoft/RD-Agent/actions/workflows/pr.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/pr.yml)
[![Readthedocs Preview](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/readthedocs-preview.yml)
[![Release.yml](https://github.com/microsoft/RD-Agent/actions/workflows/release.yml/badge.svg)](https://github.com/microsoft/RD-Agent/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/rdagent)](https://pypi.org/project/rdagent/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rdagent)](https://pypi.org/project/rdagent/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- [![Release](https://img.shields.io/github/v/release/microsoft/RD-Agent)](https://github.com/microsoft/RD-Agent/releases) -->
<!-- [![GitHub](https://img.shields.io/github/license/microsoft/RD-Agent)](https://github.com/microsoft/RD-Agent/blob/main/LICENSE) -->

<!-- Comment badge to wait until the project is publicly available. --> 

# 📰 News
| 🗞️News        | 📝Description                 |
| --            | ------                        |
| First release | RDAgent is release on Github |


# 🌟 Introduction

![Our focused scenario](docs/_static/scen.jpg)

RDAgent aims to automate the most critical and valuable aspects of the industrial R&D process, and we begin with focusing on the data-driven scenarios to streamline the development of models and data. 
Methodologically, we have identified a framework with two key components: 'R' for proposing new ideas and 'D' for implementing them.
We believe that the automatic evolution of R&D will lead to solutions of significant industrial value.


<!-- Tag Cloud -->
R&D is a very general scenario. The advent of RDAgent can be your
- [🎥Automatic Quant Factory](https://rdagent.azurewebsites.net/factor_loop)
- 🤖Data mining agent: iteratively proposing [🎥data](https://rdagent.azurewebsites.net/dmm) & [models](https://rdagent.azurewebsites.net/model_loop) and implementing them by gaining knowledge from data.
- 🦾Research copilot: Auto read [🎥research papers](https://rdagent.azurewebsites.net/report_model)/[🎥reports](https://rdagent.azurewebsites.net/report_factor) and implement model structures or building datasets.
- ...

You can click the [🎥link](https://rdagent.azurewebsites.net) above to view the demo. More methods and scenarios are being added to the project to empower your R&D processes and boost productivity.
<!-- 
- TODO: Demo: it fails to display the video in the README.md.
We have a quick 🎥demo for one use case of RDAgent.
[![Demo Video](https://img.youtube.com/vi/5275fcb75803ad2bb9541c3abd86dedfd578a28fa32b46fa28917b33/0.jpg)](https://rdagent.azurewebsites.net:443/media/5275fcb75803ad2bb9541c3abd86dedfd578a28fa32b46fa28917b33.mp4)
 <p align="center">
  <a href="https://rdagent.azurewebsites.net/"> <img src="docs/_static/img/logo/1.png" /> </a>
 </p>
 -->
 
 [![Watch the video](https://img.freepik.com/premium-vector/video-streaming-media-player-template-mockup-live-stream-window-player-online-broadcasting_659151-73.jpg)](https://rdagent.azurewebsites.net/)
 

# ⚡Quick start
You can try above demos by running the following command:

### 🐳Docker installation.
Users must ensure Docker is installed before attempting most scenarios. Please refer to the [official 🐳Docker page](https://docs.docker.com/engine/install/) for installation instructions.

### 🐍 Create a Conda Environment
- Create a new conda environment with Python (3.10 and 3.11 are well-tested in our CI):
  ```sh
  conda create -n rdagent python=3.10
  ```
- Activate the environment:
  ```sh
  conda activate rdagent
  ```

### 🛠️ Install the RDAgent
- You can directly install the RDAgent package from PyPI:
  ```sh
  pip install rdagent
  ```

### ⚙️ Configuration
You have to config your GPT model in the `.env`
```bash
cat << EOF  > .env
OPENAI_API_KEY=<your_api_key>
# EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4-turbo
EOF
```

### 🚀 Run the Application

The [🎥demo](https://rdagent.azurewebsites.net) is implemented by the following commands(each item represents one demo, you can select the one you prefer):

- Run the **Automated Quantitative Trading & Iterative Factors Evolution**:  Qlib self-loop factor proposal and implementation application
  ```sh
  rdagent fin_factor
  ```

- Run the **Automated Quantitative Trading & Iterative Model Evolution**: Qlib self-loop model proposal and implementation application
  ```sh
  python fin_model
  ```

- Run the **Automated Medical Predtion Model Evolution**: medical self-loop model proposal and implementation application
  ```sh
  python med_model
  ```

- Run the **Automated Quantitative Trading & Factors Extraction from Financial Reports**:  Run the Qlib factor extraction and implementation application based on financial reports
  ```sh
  rdagent fin_factor_report <Your report folder>
  ```

- Run the **Automated Model Research & Development Co-Pilot**: model extraction and implementation application
  ```sh
  rdagent general_model  <Your paper url>
  ```

### 🚀 Monitor the Application Results
- You can serve our demo app to monitor the RD loop by running the following command:
  ```sh
  rdagent ui --port 80 --log_dir <your log folder like "log/2024-07-16_11-21-46-612120/">
  ```

# Scenarios

We have applied RD-Agent to multiple valuable data-driven industrial scenarios..


## 🎯 Goal: Agent for Data-driven R&D

In this project, we are aiming to build a Agent to automate Data-Driven R\&D that can
+ 📄Read real-world material (reports, papers, etc.) and **extract** key formulas, descriptions of interested **features** and **models**, which are the key components of data-driven R&D .
+ 🛠️**Implement** the extracted formulas (e.g., features, factors, and models) in runnable codes.
   + Due to the limited ability of LLM in implementing at once, evolve the agent to be able to extend abilities by learning from feedback and knowledge and improve the agent's ability to implement more complex models.
+ 💡Propose **new ideas** based on current knowledge and observations.

<!-- ![Data-Centric R&D Overview](docs/_static/overview.png) -->

## 📈 Scenarios/Demos

In the two key areas of data-driven scenarios, model implementation and data building, our system aims to serve two main roles: 🦾copilot and 🤖agent. 
- The 🦾copilot follows human instructions to automate repetitive tasks. 
- The 🤖agent, being more autonomous, actively proposes ideas for better results in the future.

The supported scenarios are listed below:

| Scenario/Target | Model Implementation                   | Data Building                                                                      |
| --              | --                                     | --                                                                                 |
| 💹 Finance      | 🤖[Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/model_loop) | - 🦾[Auto reports reading & implementation](https://rdagent.azurewebsites.net/report_factor) <br/> - 🤖[Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/factor_loop) |
| 🩺 Medical      | 🤖[Iteratively Proposing Ideas & Evolving](https://rdagent.azurewebsites.net/dmm) | -                                                                                  |
| 🏭 General      | 🦾[Auto paper reading & implementation](https://rdagent.azurewebsites.net/report_model)    | -                                                                                  |

Different scenarios vary in entrance and configuration. Please check the detailed setup tutorial in the scenarios documents.

Here is a gallery of successful explorations. You can download the source code and view the execution trace using the command below:

```bash
rdagent ui --port 80 --log_dir gallary/
```


# ⚙️Framework

![image](https://github.com/user-attachments/assets/98fce923-77ab-4982-93c8-a7a01aece766)


Automating the R&D process in data science is a highly valuable yet underexplored area in industry. We propose a framework to push the boundaries of this important research field.

The research questions within this framework can be divided into three main categories:
| Research Area | Paper/Work List |
|--------------------|-----------------|
| Benchmark the R&D abilities | [Benchmark](#benchmark) |
| Idea proposal: Explore new ideas or refine existing ones | [Research](#research) |
| Ability to realize ideas: Implement and execute ideas | [Development](#development) |

We believe that the key to delivering high-quality solutions lies in the ability to evolve R&D capabilities. Agents should learn like human experts, continuously improving their R&D skills.


# 📃Paper/Work list

## Benchmark
- [Towards Data-Centric Automatic R&D](https://arxiv.org/abs/2404.11276);
```BibTeX
@misc{chen2024datacentric,
    title={Towards Data-Centric Automatic R&D},
    author={Haotian Chen and Xinjie Shen and Zeqi Ye and Wenjun Feng and Haoxue Wang and Xiao Yang and Xu Yang and Weiqing Liu and Jiang Bian},
    year={2024},
    eprint={2404.11276},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
![image](https://github.com/user-attachments/assets/494f55d3-de9e-4e73-ba3d-a787e8f9e841)

## Research

In a data mining expert's daily research and development process, they propose a hypothesis (e.g., a model structure like RNN can capture patterns in time-series data), design experiments (e.g., finance data contains time-series and we can verify the hypothesis in this scenario), implement the experiment as code (e.g., Pytorch model structure), and then execute the code to get feedback (e.g., metrics, loss curve, etc.). The experts learn from the feedback and improve in the next iteration.

Based on the principles above, we have established a basic method framework that continuously proposes hypotheses, verifies them, and gets feedback from the real-world practice. This is the first scientific research automation framework that supports linking with real-world verification.

For more detail, please refer to our [Demos page](https://rdagent.azurewebsites.net).

## Development

- [Collaborative Evolving Strategy for Automatic Data-Centric Development](https://arxiv.org/abs/2407.18690)
```BibTeX
@misc{yang2024collaborative,
    title={Collaborative Evolving Strategy for Automatic Data-Centric Development},
    author={Xu Yang and Haotian Chen and Wenjun Feng and Haoxue Wang and Zeqi Ye and Xinjie Shen and Xiao Yang and Shizhao Sun and Weiqing Liu and Jiang Bian},
    year={2024},
    eprint={2407.18690},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
![image](https://github.com/user-attachments/assets/75d9769b-0edd-4caf-9d45-57d1e577054b)


# Contributing

More documents can be found in the [📚readthedocs](https://rdagent.readthedocs.io/).

## Guidance
This project welcomes contributions and suggestions.
You can find issues in the issues list or simply running `grep -r "TODO:"`.

Making contributions is not a hard thing. Solving an issue(maybe just answering a question raised in issues list ), fixing/issuing a bug, improving the documents and even fixing a typo are important contributions to RDAgent.
<img src="https://img.shields.io/github/contributors-anon/microsoft/RD-Agent"/>

<a href="https://github.com/microsoft/RD-Agent/graphs/contributors"><img src="https://contrib.rocks/image?repo=microsoft/RD-Agent" /></a>

# Legal disclaimer
<p style="line-height: 1; font-style: italic;">The RD-agent is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. The RD-agent is aimed to facilitate research and development process in the financial industry and not ready-to-use for any financial investment or advice. Users shall independently assess and test the risks of the RD-agent in a specific use scenario, ensure the responsible use of AI technology, including but not limited to developing and integrating risk mitigation measures, and comply with all applicable laws and regulations in all applicable jurisdictions. The RD-agent does not provide financial opinions or reflect the opinions of Microsoft, nor is it designed to replace the role of qualified financial professionals in formulating, assessing, and approving finance products. The inputs and outputs of the RD-agent belong to the users and users shall assume all liability under any theory of liability, whether in contract, torts, regulatory, negligence, products liability, or otherwise, associated with use of the RD-agent and any inputs and outputs thereof.</p>
