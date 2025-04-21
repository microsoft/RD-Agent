.. _data_science_agent:

=======================
Data Science Agent
=======================

**🤖 Automated Feature Engineering & Model Tuning Evolution**
------------------------------------------------------------------------------------------

🎨 Design
~~~~~~~~~~~

.. image:: kaggle_design.png
   :alt: Design of Data Science Agent
   :align: center

📖 Background
~~~~~~~~~~~~~~
In the landscape of data science competitions, Data Science serves as the ultimate arena where data enthusiasts harness the power of algorithms to tackle real-world challenges.
The Data Science Agent stands as a pivotal tool, empowering participants to seamlessly integrate cutting-edge models and datasets, transforming raw data into actionable insights.

By utilizing the **Data Science Agent**, data scientists can craft innovative solutions that not only uncover hidden patterns but also drive significant advancements in predictive accuracy and model robustness.


🌟 Introduction
~~~~~~~~~~~~~~~~

In this scenario, our automated system proposes hypothesis, choose action, implements code, conducts validation, and utilizes feedback in a continuous, iterative process.

The goal is to automatically optimize performance metrics within the validation set or Data Science Leaderboard, ultimately discovering the most efficient features and models through autonomous research and development.

Here's an enhanced outline of the steps:

**Step 1 : Hypothesis Generation 🔍**

- Generate and propose initial hypotheses based on previous experiment analysis and domain expertise, with thorough reasoning and financial justification.

**Step 2 : Experiment Creation ✨**

- Transform the hypothesis into a task.
- Choose a specific action within feature engineering or model tuning.
- Develop, define, and implement a new feature or model, including its name, description, and formulation.

**Step 3 : Model/Feature Implementation 👨‍💻**

- Implement the model code based on the detailed description.
- Evolve the model iteratively as a developer would, ensuring accuracy and efficiency.

**Step 4 : Validation on Test Set or Data Science 📉**

- Validate the newly developed model using the test set or Data Science dataset.
- Assess the model's effectiveness and performance based on the validation results.

**Step 5: Feedback Analysis 🔍**

- Analyze validation results to assess performance.
- Use insights to refine hypotheses and enhance the model.

**Step 6: Hypothesis Refinement ♻️**

- Adjust hypotheses based on validation feedback.
- Iterate the process to continuously improve the model.

🧭 Example Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 🔧 **Set up RD-Agent Environment**

  - Before you start, please make sure you have installed RD-Agent and configured the environment for RD-Agent correctly. If you want to know how to install and configure the RD-Agent, please refer to the `documentation <../installation_and_configuration.html>`_.

- 🔩 **Setting the Environment variables at .env file**

  - Determine the path where the data will be stored and add it to the ``.env`` file.

  .. code-block:: sh

    dotenv set KG_LOCAL_DATA_PATH <your local directory>/kaggle_data
    dotenv set DS_SCEN rdagent.scenarios.data_science.scen.DataScienceScen

- 📥 **Prepare Competition Data**

  - Data Science competition data, contains three parts: competition description file (markdown file), competition dataset and competition evaluation files.

    - **Correct directory structure (Here is an example of competition data with id custom_data)**

      .. code-block:: text

        kaggle_data
        └── eval
        | └── custom_data
        |    └── grade.py
        |    └── valid.py
        |    └── test.csv
        └── custom_data
          └── train.csv
          └── test.csv
          └── sample_submission.csv
          └── description.md
        
      - ``kaggle_data/custom_data/train.csv:`` Necessary training data in csv or parquet format, or training images.

      - ``kaggle_data/custom_data/description.md:`` (Optional) Competition description file.

      - ``kaggle_data/custom_data/sample_submission.csv:`` (Optional) Competition sample submission file.

      - ``kaggle_data/eval/custom_data/grade.py:`` (Optional) Competition grade script, in order to calculate the score for the submission.

      - ``kaggle_data/eval/custom_data/valid.py:`` (Optional) Competition validation script, in order to check if the submission format is correct.

      - ``kaggle_data/eval/custom_data/submission_test.csv:`` (Optional) Competition test label file.


- 🚀 **Run the Application**

  - You can directly run the application by using the following command:
    
    .. code-block:: sh

        rdagent kaggle --competition <Competition ID>


🎨 Customize one template for a new competition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In order to facilitate RD-Agent to generate competition codes, we have specified a competition code structure:

.. image:: kaggle_template.png
   :alt: Design of Data Science Code Template
   :align: center

- **feature directory** contains the feature engineering code. Generally no modification is required.
- **model directory** contains the model codes.
  select_xx.py is used to select different features according to different models.
  model_xx.py is the basic code of different models. Generally, only some initial parameters need to be adjusted.
- **fea_share_preprocess.py** is some basic preprocessing code shared by different models. The degree of customization here is high, but the preprocess_script() function needs to be retained, which will be called by train.py
- **train.py** is the main code, which connects all the codes and is also the code called during the final execution.

**We will soon provide a tool for automatic/semi-automatic template generation.**
If you want to try a different competition now, you can refer to our current template structure and content to write a new template.


🎯 Roadmap
~~~~~~~~~~~

**Completed:**

- **Data Science Project Schema Design** ✅

- **RD-Agent Integration with kaggle schema** ✅

**Ongoing:**

- **Template auto generation**

- **Bench Optimization**

  - **Online Bench**

    - **RealMLBench**

      - Ongoing integration

      - Auto online submission

      - Batch Evaluation

  - **Offline Bench**
  
    - MLE-Bench


🛠️ Usage of modules
~~~~~~~~~~~~~~~~~~~~~

.. _Env Config: 

- **Env Config**

The following environment variables can be set in the `.env` file to customize the application's behavior:

.. autopydantic_settings:: rdagent.app.kaggle.conf.Data ScienceBasePropSetting
    :settings-show-field-summary: False
    :exclude-members: Config

.. autopydantic_settings:: rdagent.components.coder.factor_coder.config.FactorCoSTEERSettings
    :settings-show-field-summary: False
    :members: coder_use_cache, file_based_execution_timeout, select_method, max_loop
    :exclude-members: Config, fail_task_trial_limit, v1_query_former_trace_limit, v1_query_similar_success_limit, v2_query_component_limit, v2_query_error_limit, v2_query_former_trace_limit, v2_error_summary, v2_knowledge_sampler, v2_add_fail_attempt_to_latest_successful_execution, new_knowledge_base_path, knowledge_base_path, data_folder, data_folder_debug
    :no-index:
