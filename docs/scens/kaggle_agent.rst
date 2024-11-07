.. _kaggle_agent:

=======================
Kaggle Agent
=======================

**🤖 Automated Feature Engineering & Model Tuning Evolution**
------------------------------------------------------------------------------------------

🎨 Design
~~~~~~~~~~~

.. image:: kaggle_design.png
   :alt: Design of Kaggle Agent
   :align: center

📖 Background
~~~~~~~~~~~~~~
In the landscape of data science competitions, Kaggle serves as the ultimate arena where data enthusiasts harness the power of algorithms to tackle real-world challenges.
The Kaggle Agent stands as a pivotal tool, empowering participants to seamlessly integrate cutting-edge models and datasets, transforming raw data into actionable insights.

By utilizing the **Kaggle Agent**, data scientists can craft innovative solutions that not only uncover hidden patterns but also drive significant advancements in predictive accuracy and model robustness.


🌟 Introduction
~~~~~~~~~~~~~~~~

In this scenario, our automated system proposes hypothesis, choose action, implements code, conducts validation, and utilizes feedback in a continuous, iterative process.

The goal is to automatically optimize performance metrics within the validation set or Kaggle Leaderboard, ultimately discovering the most efficient features and models through autonomous research and development.

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

**Step 4 : Validation on Test Set or Kaggle 📉**

- Validate the newly developed model using the test set or Kaggle dataset.
- Assess the model's effectiveness and performance based on the validation results.

**Step 5: Feedback Analysis 🔍**

- Analyze validation results to assess performance.
- Use insights to refine hypotheses and enhance the model.

**Step 6: Hypothesis Refinement ♻️**

- Adjust hypotheses based on validation feedback.
- Iterate the process to continuously improve the model.

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

        rdagent kaggle --competition [your competition name]

    The `competition name` parameter must match the name used with the API on the Kaggle platform.

(NOTE: The code for crawling Kaggle competition information may not be applicable to your environment. 

If you cannot execute it normally, you can refer to the following **Example Guide: Running a Specific Experiment**.)


📋 Competition List Available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+------------------+-----------+-------------------------------+
| **Competition Name**              | **Task**         | **Modal** | **ID**                        |
+===================================+==================+===========+===============================+
| Media Campaign Cost Dataset       | Regression       | Tabular   | playground-series-s3e11       |
+-----------------------------------+------------------+-----------+-------------------------------+
| Wild Blueberry Yield Dataset      | Regression       | Tabular   | playground-series-s3e14       |
+-----------------------------------+------------------+-----------+-------------------------------+
| Crab Age Dataset                  | Regression       | Tabular   | playground-series-s3e16       |
+-----------------------------------+------------------+-----------+-------------------------------+
| Flood Prediction Dataset          | Regression       | Tabular   | playground-series-s4e5        |
+-----------------------------------+------------------+-----------+-------------------------------+
| Used Car Prices Dataset           | Regression       | Tabular   | playground-series-s4e9        |
+-----------------------------------+------------------+-----------+-------------------------------+
| Cirrhosis Outcomes Dataset        | Multi-Class      | Tabular   | playground-series-s3e26       |
+-----------------------------------+------------------+-----------+-------------------------------+
| San Francisco Crime Classification| Multi-Class      | Tabular   | sf-crime                      |
+-----------------------------------+------------------+-----------+-------------------------------+
| Poisonous Mushrooms Dataset       | Classification   | Tabular   | playground-series-s4e8        |
+-----------------------------------+------------------+-----------+-------------------------------+
| Spaceship Titanic                 | Classification   | Tabular   | spaceship-titanic             |
+-----------------------------------+------------------+-----------+-------------------------------+
| Forest Cover Type Prediction      | Classification   | Tabular   | forest-cover-type-prediction  |
+-----------------------------------+------------------+-----------+-------------------------------+
| Digit Recognizer                  | Classification   | Image     | digit-recognizer              |
+-----------------------------------+------------------+-----------+-------------------------------+
| To be continued ...                                                                              |
+-----------------------------------+------------------+-----------+-------------------------------+



🧭 Example Guide: Running a Specific Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Example Competition Name**: San Francisco Crime Classification (sf-crime)

- 🔧 **Set up RD-Agent Environment**

- 📥 **Download Competition Data (Optional)** 

  - If web information and data are not prepared locally, the RD-Agent loop will automatically download them at startup. 

  - In this case, you need to configure the Kaggle API yourself, agree to the competition rules on the Kaggle page, and configure `chromedriver` for Selenium. 

  - Alternatively, you can manually place the dataset in the specified location in advance.

  - you can download the kaggle_data.zip in the release and unzip it to the directory configured by the `KG_LOCAL_DATA_PATH` environment variable. 


- 🚀 **Run the Application**

  - You can directly run the application by using the following command:
    
    .. code-block:: sh

        rdagent kaggle --competition sf-crime

- 📤 **Submit the Result Automatically or Manually**

  - If Auto: You need to configure the Kaggle API, agree to the competition rules on the page, and set `KG_AUTO_SUBMIT=true`.
  
  - Else: You can download the prediction results from the UI interface and submit them manually. For more details, refer to the :doc:`UI guide <../ui>`.

For more information about Kaggle API Settings, refer to the `Kaggle API <https://github.com/Kaggle/kaggle-api>`_.


🎨 Customize one template for a new competition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In order to facilitate RD-Agent to generate competition codes, we have specified a competition code structure:

.. image:: kaggle_template.png
   :alt: Design of Kaggle Code Template
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

- **Kaggle Project Schema Design** ✅

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

.. autopydantic_settings:: rdagent.app.kaggle.conf.KaggleBasePropSetting
    :settings-show-field-summary: False
    :exclude-members: Config

.. autopydantic_settings:: rdagent.components.coder.factor_coder.config.FactorImplementSettings
    :settings-show-field-summary: False
    :members: coder_use_cache, data_folder, data_folder_debug, file_based_execution_timeout, select_method, select_threshold, max_loop, knowledge_base_path, new_knowledge_base_path
    :exclude-members: Config, fail_task_trial_limit, v1_query_former_trace_limit, v1_query_similar_success_limit, v2_query_component_limit, v2_query_error_limit, v2_query_former_trace_limit, v2_error_summary, v2_knowledge_sampler, v2_add_fail_attempt_to_latest_successful_execution, new_knowledge_base_path, knowledge_base_path, data_folder, data_folder_debug, select_threshold
    :no-index:
