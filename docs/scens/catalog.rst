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
      - :ref:`🥇The First Data-Centric Quant Multi-Agent Framework <quant_agent_fin>`
      - :ref:`🤖Iteratively Proposing Ideas & Evolving <model_agent_fin>`  
        
        :ref:`🦾Auto reports reading & implementation <data_copilot_fin>`  
        
        :ref:`🤖Iteratively Proposing Ideas & Evolving <data_agent_fin>`
    * - 🏭 General
      - :ref:`🦾Auto paper reading & implementation <model_copilot_general>`  
        
        :ref:`🧪FT-Agent for LLM fine-tuning <finetune_agent>`
      - :ref:`🤖 Data Science <data_science_agent>`


.. toctree::
    :maxdepth: 1
    :caption: Doctree:
    :hidden:

    quant_agent_fin
    data_agent_fin
    data_copilot_fin
    model_agent_fin
    model_copilot_general
    data_science
    finetune
