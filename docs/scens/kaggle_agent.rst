.. _kaggle_agent:

=======================
Kaggle Agent
=======================

**🤖 Automated Feature Engineering & Model Tuning Evolution**
------------------------------------------------------------------------------------------

📖 Background
~~~~~~~~~~~~~~
In the landscape of data science competitions, Kaggle serves as the ultimate arena where data enthusiasts harness the power of algorithms to tackle real-world challenges.
The Kaggle Agent stands as a pivotal tool, empowering participants to seamlessly integrate cutting-edge models and datasets, transforming raw data into actionable insights.

By utilizing the **Kaggle Agent**, data scientists can craft innovative solutions that not only uncover hidden patterns but also drive significant advancements in predictive accuracy and model robustness.


🎥 `Demo <https://rdagent.azurewebsites.net/kaggle_loop>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div style="display: flex; justify-content: center; align-items: center;">
      <video width="600" controls>
        <source src="https://rdagent.azurewebsites.net/media/??????.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>

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

- Set up your config

  - Set up your local_data_path:

    .. code-block:: python

        local_data_path: str = "path/to/your/data/location"

- 🚀 Run the Application

  - You can directly run the application by using the following command:
    
    .. code-block:: sh

        python3 rdagent/app/kaggle/loop.py --competition [your competition name]

🛠️ Usage of modules
~~~~~~~~~~~~~~~~~~~~~

.. _Env Config: 

- **Env Config**

The following environment variables can be set in the `.env` file to customize the application's behavior:

.. autopydantic_settings:: rdagent.app.kaggle.conf.KaggleBasePropSetting
    :settings-show-field-summary: False
    :exclude-members: Config

.. autopydantic_settings:: rdagent.components.coder.factor_coder.config.FactorImplementSettings
    :settings-show-field-summary: False
    :members: coder_use_cache, data_folder, data_folder_debug, file_based_execution_timeout, select_method, select_threshold, max_loop, knowledge_base_path, new_knowledge_base_path
    :exclude-members: Config, fail_task_trial_limit, v1_query_former_trace_limit, v1_query_similar_success_limit, v2_query_component_limit, v2_query_error_limit, v2_query_former_trace_limit, v2_error_summary, v2_knowledge_sampler, v2_add_fail_attempt_to_latest_successful_execution
    :no-index:


