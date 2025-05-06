=========================
Scenarios
=========================

Scenario lists
=========================

In the two key areas of data-driven scenarios, model implementation and data building, our system aims to serve two main roles: 🦾copilot and 🤖agent.

- The 🦾copilot follows human instructions to automate repetitive tasks.
- The 🤖agent, being more autonomous, actively proposes ideas for better results in the future.

The supported scenarios are listed below:



.. list-table:: 
    :header-rows: 1

    * - Scenario/Target
      - Model Implementation
      - Data Building
    * - 💹 Finance
      - :ref:`🤖Iteratively Proposing Ideas & Evolving <model_agent_fin>`
      - :ref:`🦾Auto reports reading & implementation <data_copilot_fin>`

        :ref:`🤖Iteratively Proposing Ideas & Evolving <data_agent_fin>`
    * - 🩺 Medical
      - :ref:`🤖Iteratively Proposing Ideas & Evolving <model_agent_med>`
      - 
    * - 🏭 General
      - :ref:`🦾Auto paper reading & implementation <model_copilot_general>`

        :ref:`🤖Auto Kaggle Model Tuning <kaggle_agent>`
      - :ref:`🤖Auto Kaggle feature Engineering <kaggle_agent>`


.. toctree::
    :maxdepth: 1
    :caption: Doctree:
    :hidden:

    data_agent_fin
    data_copilot_fin
    model_agent_fin
    model_agent_med
    model_copilot_general
    kaggle_agent
    data_science

