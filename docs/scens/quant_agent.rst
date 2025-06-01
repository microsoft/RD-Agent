.. _quant_agent_fin:

=====================
Finance Quant Agent
=====================


**🥇The First Data-Centric Quant Multi-Agent Framework RD-Agent(Q)**
---------------------------------------------------------------------

⚡ Quick Start
~~~~~~~~~~~~~~~~~

Please refer to the installation part in :doc:`../installation_and_configuration` to prepare your system dependency.

You can try our demo by running the following command:

- 🐍 Create a Conda Environment

  - Create a new conda environment with Python (3.10 and 3.11 are well tested in our CI):

    .. code-block:: sh

          conda create -n rdagent python=3.10

  - Activate the environment:

    .. code-block:: sh

        conda activate rdagent

- 📦 Install the RDAgent
  
  - You can install the RDAgent package from PyPI:

    .. code-block:: sh

        pip install rdagent

- 🚀 Run the Application
    
  - You can directly run the application by using the following command:
    
    .. code-block:: sh

        rdagent fin_quant


🛠️ Usage of modules
~~~~~~~~~~~~~~~~~~~~~

.. _Env Config: 

- **Env Config**

The following environment variables can be set in the `.env` file to customize the application's behavior:

.. autopydantic_settings:: rdagent.app.qlib_rd_loop.conf.QuantBasePropSetting
    :settings-show-field-summary: False
    :exclude-members: Config

.. autopydantic_settings:: rdagent.components.coder.factor_coder.config.FactorCoSTEERSettings
    :settings-show-field-summary: False
    :members: coder_use_cache, data_folder, data_folder_debug, file_based_execution_timeout, select_method, max_loop, knowledge_base_path, new_knowledge_base_path
    :exclude-members: Config, fail_task_trial_limit, v1_query_former_trace_limit, v1_query_similar_success_limit, v2_query_component_limit, v2_query_error_limit, v2_query_former_trace_limit, v2_error_summary, v2_knowledge_sampler
    :no-index:

