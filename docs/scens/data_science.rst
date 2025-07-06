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

  - A data science competition dataset usually consists of two parts: ``competition dataset`` and ``evaluation dataset``. (We provide `a sample <https://github.com/microsoft/RD-Agent/tree/main/rdagent/scenarios/data_science/example>`_ of a customized dataset named: `arf-12-hour-prediction-task as a reference`.)
    
    - The ``competition dataset`` contains **training data**, **test data**, **description files**, **formatted submission files**, **data sampling codes**.
    
    - The ``evaluation dataset`` contains **standard answer file**, **data checking codes**, and **Code for calculation of scores**.

  - We use the ``arf-12-hour-prediction-task`` data as a sample to introduce the preparation workflow for the competition dataset.
  
    - Create a ``ds_data/source_data/arf-12-hours-prediction-task`` folder, which will be used to store your raw dataset.

      - The raw files for the competition ``arf-12-hours-prediction-task`` have two files: ``ARF_12h.csv`` and ``X.npz``.
    
    - Create a ``ds_data/source_data/arf-12-hours-prediction-task/prepare.py`` file that splits your raw data into **training data**, **test data**, **formatted submission file**, and **standard answer file**. (You will need to write a script based on your raw data.)
      
      - The following shows the preprocessing code for the raw data of ``arf-12-hours-prediction-task``.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/prepare.py
        :language: python
        :caption: ds_data/source_data/arf-12-hours-prediction-task/prepare.py
        :linenos:

      - At the end of program execution, the ``ds_data`` folder structure will look like this:

      .. code-block:: text

        ds_data
        ├── arf-12-hours-prediction-task
        │   ├── train
        │   │   ├── ARF_12h.csv
        │   │   └── X.npz
        │   ├── test
        │   │   ├── ARF_12h.csv
        │   │   └── X.npz
        │   └── sample_submission.csv
        ├── eval
        │   └── arf-12-hours-prediction-task
        │       └── submission_test.csv
        └── source_data
            └── arf-12-hours-prediction-task
                ├── ARF_12h.csv
                ├── prepare.py
                └── X.npz

    - Create a ``ds_data/arf-12-hours-prediction-task/description.md`` file to describe your competition, Objective, dataset, and other information.

      - The following shows the description file for ``arf-12-hours-prediction-task``

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/arf-12-hour-prediction-task/description.md
        :language: markdown
        :caption: ds_data/arf-12-hours-prediction-task/description.md
        :linenos:

    - Create a ``ds_data/arf-12-hours-prediction-task/sample.py`` file to construct the debugging sample data.

      - The following shows the script for constructing the debugging sample data based on the ``arf-12-hours-prediction-task`` dataset implementation.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/arf-12-hour-prediction-task/sample.py
        :language: markdown
        :caption: ds_data/arf-12-hours-prediction-task/sample.py
        :linenos:

    - Create a ``ds_data/eval/arf-12-hour-prediction-task/valid.py`` file, which is used to check the validity of the submission files to ensure that their formatting is consistent with the reference file.

      - The following shows a script that checks the validity of a submission based on the ``arf-12-hours-prediction-task`` data.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/eval/arf-12-hour-prediction-task/valid.py
        :language: markdown
        :caption: ds_data/eval/arf-12-hour-prediction-task/valid.py
        :linenos:

    - Create a ``ds_data/eval/arf-12-hour-prediction-task/grade.py`` file, which is used to calculate the score based on the submission file and the **standard answer file**, and output the result in JSON format.

      - The following shows a grading script based on the ``arf-12-hours-prediction-task`` data implementation.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/eval/arf-12-hour-prediction-task/grade.py
        :language: markdown
        :caption: ds_data/eval/arf-12-hour-prediction-task/grade.py
        :linenos:

  - At this point, you have created a complete dataset. The correct structure of the dataset should look like this.

  .. code-block:: text

      ds_data
      ├── arf-12-hours-prediction-task
      │   ├── train
      │   │   ├── ARF_12h.csv
      │   │   └── X.npz
      │   ├── test
      │   │   ├── ARF_12h.csv
      │   │   └── X.npz
      │   ├── description.md
      │   ├── sample_submission.csv
      │   └── sample.py
      ├── eval
      │   └── arf-12-hours-prediction-task
      │       ├── grade.py
      │       ├── submission_test.csv
      │       └── valid.py
      └── source_data
          └── arf-12-hours-prediction-task
              ├── ARF_12h.csv
              ├── prepare.py
              └── X.npz

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
          └── sample.py
        
      - ``ds_data/custom_data/train.csv:`` Necessary training data in csv or parquet format, or training images.

      - ``ds_data/custom_data/test.csv:`` Necessary test data in csv or parquet format, or test images.

      - ``ds_data/custom_data/description.md:`` (Optional) Competition description file.

      - ``ds_data/custom_data/sample_submission.csv:`` (Optional) Competition sample submission file.

      - ``ds_data/custom_data/sample.py:`` (Optional) Sample code for generating debug data from the competition dataset. If not provided, R&D-Agent will use its default sampling logic. For details, see the ``create_debug_data`` function in ``rdagent/scenarios/data_science/debug/data.py``.

      - ``ds_data/eval/custom_data/grade.py:`` (Optional) Competition grade script, in order to calculate the score for the submission.

      - ``ds_data/eval/custom_data/valid.py:`` (Optional) Competition validation script, in order to check if the submission format is correct.

      - ``ds_data/eval/custom_data/submission_test.csv:`` (Optional) Competition test label file.

- 🔧 **Set up Environment for Custom User-defined Dataset**

  .. code-block:: sh

      dotenv set DS_SCEN rdagent.scenarios.data_science.scen.DataScienceScen
      dotenv set DS_LOCAL_DATA_PATH <your local directory>/ds_data
      dotenv set DS_CODER_ON_WHOLE_PIPELINE True

- 🚀 **Run the Application**

  - You can directly run the application by using the following command:
    
    .. code-block:: sh

        rdagent data_science --competition <Competition ID>

    - The following shows the command to run based on the ``arf-12-hours-prediction-task`` data

      .. code-block:: sh

          rdagent data_science --competition arf-12-hours-prediction-task

  - Then, you can run the test set score corresponding to each round of the loop.

    .. code-block:: sh

        dotenv run -- python rdagent/log/mle_summary.py grade <url_to_log>

    Here, <url_to_log> refers to the parent directory of the log folder generated during the run.

- 📥 **Visualize the R&D Process**

  - We provide a web UI to visualize the log. You just need to run:

    .. code-block:: sh

        streamlit run rdagent/log/ui/dsapp.py

  - Then you can input the log path and visualize the R&D process.

🔍 MLE-bench Guide: Running ML Engineering via MLE-bench
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 📝 **MLE-bench Overview**

  - MLE-bench is a comprehensive benchmark designed to evaluate the ML engineering capabilities of AI systems using real-world scenarios. The dataset comprises 75 Kaggle competitions. Since Kaggle does not provide held-out test sets for these competitions, the benchmark includes preparation scripts that split the publicly available training data into new training and test sets, and grading scripts are provided for each competition to accurately evaluate submission scores.

- 🔧 **Set up Environment for MLE-bench**

  - Running R&D-Agent on MLE-bench is designed for full automation. There is no need for manual downloads and data preparation. Simply set the environment variable ``DS_IF_USING_MLE_DATA`` to True.  

  - At runtime, R&D-Agent will automatically build the Docker image specified at ``rdagent/scenarios/kaggle/docker/mle_bench_docker/Dockerfile``. This image is responsible for downloading the required datasets and grading files for MLE-bench.  
  
  - Note: The first run may take longer than subsequent runs as the Docker image and data are being downloaded and set up for the first time.

    .. code-block:: sh

        dotenv set DS_LOCAL_DATA_PATH <your local directory>/ds_data
        dotenv set DS_IF_USING_MLE_DATA True

- 🔨 **Configuring the Kaggle API**

  - Downloading Kaggle competition data requires the Kaggle API. You can set up the Kaggle API by following these steps:
  
    - Register and login on the `Kaggle <https://www.kaggle.com/>`_ website.

    - Click on the avatar (usually in the top right corner of the page) -> ``Settings`` -> ``Create New Token``, A file called ``kaggle.json`` will be downloaded.

    - Move ``kaggle.json`` to ``~/.config/kaggle/``

    - Modify the permissions of the ``kaggle.json`` file.

      .. code-block:: sh

        chmod 600 ~/.config/kaggle/kaggle.json

  - For more information about Kaggle API Settings, refer to the `Kaggle API <https://github.com/Kaggle/kaggle-api>`_.


- 🔩 **Setting the Environment Variables for MLE-bench**

  - In addition to auto-downloading the benchmark data, you must also configure the runtime environment for executing the competition code.  
  - Use the environment variable ``DS_CODER_COSTEER_ENV_TYPE`` to select the execution mode:
    
    • When set to docker (the default), RD-Agent utilizes the official Kaggle Docker image (``gcr.io/kaggle-gpu-images/python:latest``) to ensure that all required packages are available.  
    • If you prefer to use a custom Docker setup, you can modify the configuration using ``DS_DOCKER_IMAGE`` or ``DS_DOCKERFILE_FOLDER_PATH``.  
    • Alternatively, if your competition work only demands basic libraries, you may set ``DS_CODER_COSTEER_ENV_TYPE`` to conda. In this mode, you must create a local conda environment named “kaggle” and pre-install the necessary packages. RD-Agent will execute the competition code within this “kaggle” conda environment.

    .. code-block:: sh

      # Configure the runtime environment: choice between 'docker' (default) or 'conda'
      dotenv set DS_CODER_COSTEER_ENV_TYPE docker

- 🚀 **Run the Application**

  - You can directly run the application by using the following command:
    
    .. code-block:: sh

        rdagent data_science --competition <Competition ID>

- 📥 **Visualize the R&D Process**

  - We provide a web UI to visualize the log. You just need to run:

    .. code-block:: sh

        streamlit run rdagent/log/ui/dsapp.py

  - Then you can input the log path and visualize the R&D process.

- **Additional Guidance**

  - **Combine different LLM Models at R&D Stage**

    - You can combine different LLM models at the R&D stage. 

    - By default, when you set environment variable ``CHAT_MODEL``, it covers both R&D stages. When customizing the model for the development stage, you can set:
    
    .. code-block:: sh

      # This example sets the model to "o3-mini". For some models, the reasoning effort shoule be set to "None".
      dotenv set LITELLM_CHAT_MODEL_MAP '{"coding":{"model":"o3-mini","reasoning_effort":"high"},"running":{"model":"o3-mini","reasoning_effort":"high"}}'




