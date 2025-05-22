.. _data_science_agent:

=======================
Data Science Agent
=======================

**🤖 Automated Feature Engineering & Model Tuning Evolution**
------------------------------------------------------------------------------------------
The Data Science Agent is an agent that can automatically perform feature engineering and model tuning. It can be used to solve various data science problems, such as image classification, time series forecasting, and text classification.

🧭 Example Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 🔧 **Set up RD-Agent Environment**

  - Before you start, please make sure you have installed RD-Agent and configured the environment for RD-Agent correctly. If you want to know how to install and configure the RD-Agent, please refer to the `documentation <../installation_and_configuration.html>`_.

- 🔩 **Setting the Environment variables at .env file**

  - Determine the path where the data will be stored and add it to the ``.env`` file.

  .. code-block:: sh

    dotenv set DS_LOCAL_DATA_PATH <your local directory>/ds_data
    dotenv set DS_SCEN rdagent.scenarios.data_science.scen.DataScienceScen

- 📥 **Prepare Competition Data**

  - Data Science competition data, contains three parts: competition description file (markdown file), competition dataset and competition evaluation files.

    - **Correct directory structure (Here is an example of competition data with id custom_data)**

      .. code-block:: text

        ds_data
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
        
      - ``ds_data/custom_data/train.csv:`` Necessary training data in csv or parquet format, or training images.

      - ``ds_data/custom_data/description.md:`` (Optional) Competition description file.

      - ``ds_data/custom_data/sample_submission.csv:`` (Optional) Competition sample submission file.

      - ``ds_data/eval/custom_data/grade.py:`` (Optional) Competition grade script, in order to calculate the score for the submission.

      - ``ds_data/eval/custom_data/valid.py:`` (Optional) Competition validation script, in order to check if the submission format is correct.

      - ``ds_data/eval/custom_data/submission_test.csv:`` (Optional) Competition test label file.



