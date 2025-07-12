.. _data_science_agent:

=======================
Data Science Agent
=======================

**🤖 Automated Feature Engineering & Model Tuning Evolution**
------------------------------------------------------------------------------------------
The Data Science Agent is an agent that can automatically perform feature engineering and model tuning. It can be used to solve various data science problems, such as image classification, time series forecasting, and text classification.

🌟 Introduction
^^^^^^^^^^^^^^^^

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

📖 Data Science Background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the evolving landscape of artificial intelligence, **Data Science** represents a powerful paradigm where machines engage in autonomous exploration, hypothesis testing, and model development across diverse domains — from healthcare and finance to logistics and research.

The **Data Science** Agent stands as a central engine in this transformation, enabling users to automate the entire machine learning workflow: from hypothesis generation to code implementation, validation, and refinement — all guided by performance feedback.

By leveraging the **Data Science** Agent, researchers and developers can accelerate experimentation cycles. Whether fine-tuning custom models or competing in high-stakes benchmarks like Kaggle, the Data Science Agent unlocks new frontiers in intelligent, self-directed discovery.

🧭 Example Guide - Customized dataset
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

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/ds_data/source_data/arf-12-hours-prediction-task/prepare.py
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

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/ds_data/arf-12-hour-prediction-task/description.md
        :language: markdown
        :caption: ds_data/arf-12-hours-prediction-task/description.md
        :linenos:

    - Create a ``ds_data/arf-12-hours-prediction-task/sample.py`` file to construct the debugging sample data.

      - The following shows the script for constructing the debugging sample data based on the ``arf-12-hours-prediction-task`` dataset implementation.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/ds_data/arf-12-hour-prediction-task/sample.py
        :language: markdown
        :caption: ds_data/arf-12-hours-prediction-task/sample.py
        :linenos:

    - Create a ``ds_data/eval/arf-12-hour-prediction-task/valid.py`` file, which is used to check the validity of the submission files to ensure that their formatting is consistent with the reference file.

      - The following shows a script that checks the validity of a submission based on the ``arf-12-hours-prediction-task`` data.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/ds_data/eval/arf-12-hour-prediction-task/valid.py
        :language: markdown
        :caption: ds_data/eval/arf-12-hour-prediction-task/valid.py
        :linenos:

    - Create a ``ds_data/eval/arf-12-hour-prediction-task/grade.py`` file, which is used to calculate the score based on the submission file and the **standard answer file**, and output the result in JSON format.

      - The following shows a grading script based on the ``arf-12-hours-prediction-task`` data implementation.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/ds_data/eval/arf-12-hour-prediction-task/grade.py
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

  - The above shows the complete dataset creation workflow, some of the files are not required, in practice you can customize the dataset according to your own needs.

    - If we don't need the test set scores, then we can choose not to generate **formatted submission files** and **standard answer file** in the prepare code, and we don't need to write **data checking codes** and **Code for calculation of scores**.

    - **data sampling codes** can also be created according to the actual need, if you do not provide **data sampling codes**, the program will use the `sampling method provided by the RD-Agent by default <https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/data_science/debug/data.py#L605>`_.

      - In the default sampling method (``create_debug_data``), the default sampling ratio (parameter: ``min_frac``) is 1%, if 1% of the data is less than 5, then 5 data will be sampled (parameter: ``min_num``), you can adjust the sampling ratio by adjusting these two parameters.

        .. code-block:: sh

          python rdagent/app/data_science/debug.py --dataset_path <dataset path> --competition <competiton_name> --min_frac <sampling ratio> --min_num <minimum number of sampling>

  - If you don't need the score of the test set, and use the sampling method provided by RD-Agent by default, then you only need to prepare a simplest dataset. The structure of the simplest dataset should look like this.

    .. code-block:: text

        ds_data
        ├── arf-12-hours-prediction-task
        │   ├── train
        │   │   ├── ARF_12h.csv
        │   │   └── X.npz
        │   ├── test
        │   │   ├── ARF_12h.csv
        │   │   └── X.npz
        │   └── description.md
        └── source_data
            └── arf-12-hours-prediction-task
                ├── ARF_12h.csv
                ├── prepare.py
                └── X.npz

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

  - 📥 **Visualize the R&D Process**

    - We provide a web UI to visualize the log. You just need to run:

      .. code-block:: sh

          streamlit run rdagent/log/ui/dsapp.py

    - Then you can input the log path and visualize the R&D process.

  - 🧪 **Scoring the test results**

    - Finally, shutdown the program, and get the test set scores with this command.

    .. code-block:: sh

      dotenv run -- python rdagent/log/mle_summary.py grade <url_to_log>

    Here, <url_to_log> refers to the parent directory of the log folder generated during the run.

🕹️ Kaggle Agent
~~~~~~~~~~~~~~~~

📖 Background
^^^^^^^^^^^^^^

In the landscape of data science competitions, Kaggle serves as the ultimate arena where data enthusiasts harness the power of algorithms to tackle real-world challenges.
The Kaggle Agent stands as a pivotal tool, empowering participants to seamlessly integrate cutting-edge models and datasets, transforming raw data into actionable insights.

By utilizing the **Kaggle Agent**, data scientists can craft innovative solutions that not only uncover hidden patterns but also drive significant advancements in predictive accuracy and model robustness.

🧭 Example Guide - Kaggle Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

📦 Preparing for the run
"""""""""""""""""""""""""

- 🔧 **Set up RD-Agent Environment**

  - Before you start, please make sure you have installed RD-Agent and configured the environment for RD-Agent correctly. If you want to know how to install and configure the RD-Agent, please refer to the `documentation <../installation_and_configuration.html>`_.

- 🔨 **Configuring the Kaggle API**

  - Register and login on the `Kaggle <https://www.kaggle.com/>`_ website.
  - Click on the avatar (usually in the top right corner of the page) -> ``Settings`` -> ``Create New Token``, A file called ``kaggle.json`` will be downloaded.
  - Move ``kaggle.json`` to ``~/.config/kaggle/``
  - Modify the permissions of the ``kaggle.json`` file.

    .. code-block:: sh

      chmod 600 ~/.config/kaggle/kaggle.json

  - For more information about Kaggle API Settings, refer to the `Kaggle API <https://github.com/Kaggle/kaggle-api>`_.

- 🔩 **Setting the Environment variables at .env file**

  - Determine the path where the data will be stored and add it to the ``.env`` file.

  .. code-block:: sh

    mkdir -p <your local directory>/ds_data
    dotenv set KG_LOCAL_DATA_PATH <your local directory>/ds_data

- 🗳️ **Join the competition**

  - If your Kaggle API account has not joined a competition, you will need to join the competition before running the program.

    - At the bottom of the competition details page, you can find the ``Join the competition`` button, click on it and select ``I Understand and Accept`` to join the competition.

    - In the **Competition List Available** below, you can jump to the competition details page.

📦 Preparing Dataset
""""""""""""""""""""""""""""""""""""

- As a subset of data science, kaggle's dataset still follows the data science format. Based on this, the kaggle dataset can be divided into two categories depending on whether or not it is supported by the mle-bench.- What is **MLE-Bench**?

  - What is MLE-Bench?

    - **MLE-Bench** is a comprehensive benchmark designed to evaluate the **machine learning engineering** capabilities of AI systems using real-world scenarios. The dataset includes multiple Kaggle competitions. Since Kaggle does not provide reserved test sets for these competitions, the benchmark includes preparation scripts for splitting publicly available training data into new training and test sets, and scoring scripts for each competition to accurately evaluate submission scores.

  - I'm running a competition Is **MLE-Bench** supported?

    - You can see all the competitions supported by **MLE-Bench** `here <https://github.com/openai/mle-bench/tree/main/mlebench/competitions>`_.

- Prepare datasets for **MLE-Bench** supported competitions.

  - You need to configure chromdriver, which is used to download the competition description file.

    - The following commands are the reference steps for configuring chromdriver

      .. code-block:: sh

        # install chrome
        wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
        sudo apt install ./google-chrome-stable_current_amd64.deb
        google-chrome --version
        # install chromedriver
        wget "https://storage.googleapis.com/chrome-for-testing-public/$(google-chrome --version | grep -oP '\d+\.\d+\.\d+\.\d+')/linux64/chromedriver-linux64.zip"
        unzip chromedriver-linux64.zip
        cd chromedriver-linux64
        sudo mv chromedriver /usr/local/bin
        sudo chmod +x /usr/local/bin/chromedriver
        chromedriver --version

  - Configure environment variables, add ``DS_IF_USING_MLE_DATA`` to environment variables, and set it to ``True``.

    .. code-block:: sh

      dotenv set DS_IF_USING_MLE_DATA True

  - At this point, you are ready to start running your competition, which will automatically download the data and construct a minimum dataset.

    - After running the program the structure of the ds_data folder should look like this (Using the ``tabular-playground-series-dec-2021`` contest as an example).

      .. code-block:: text

        ds_data
        ├── tabular-playground-series-dec-2021
        │   ├── description.md
        │   ├── sample_submission.csv
        │   ├── test.csv
        │   └── train.csv
        ├── sample
        │   └── tabular-playground-series-dec-2021
        │       ├── description.md
        │       ├── sample_submission.csv
        │       ├── test.csv
        │       └── train.csv
        └── zip_files
            └── tabular-playground-series-dec-2021
                └── tabular-playground-series-dec-2021.zip

  - At runtime, RD-Agent will automatically build the Docker image specified at `rdagent/scenarios/kaggle/docker/mle_bench_docker/Dockerfile <https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/kaggle/docker/mle_bench_docker/Dockerfile>`_. This image is responsible for downloading the required datasets and grading files for MLE-Bench.

  Note: The first run may take longer than subsequent runs as the Docker image and data are being downloaded and set up for the first time.

- Prepare datasets for competitions that are not supported by **MLE-Bench**.

  - As a subset of data science, we can follow the format and steps of data science dataset to prepare kaggle dataset. Below we will describe the workflow for preparing a kaggle dataset using the competition ``playground-series-s4e9`` as an example.
  
    - Create a ``ds_data/source_data/playground-series-s4e9`` folder, which will be used to store your raw dataset.

      - The raw files for the competition ``playground-series-s4e9`` have two files: ``train.csv``, ``test.csv``, ``sample_submission.csv``, and there are two ways to get the raw data:

        - You can find the raw data required for the competition on the `official kaggle website <https://www.kaggle.com/competitions/playground-series-s4e9/data>`_.

        - Or you can use the command line to download the raw data for the competition, the download command is as follows.

          .. code-block:: sh

            kaggle competitions download -c playground-series-s4e9

    - Create a ``ds_data/source_data/playground-series-s4e9/prepare.py`` file that splits your raw data into **training data**, **test data**, **formatted submission file**, and **standard answer file**. (You will need to write a script based on your raw data.)
      
      - The following shows the preprocessing code for the raw data of ``playground-series-s4e9``.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/ds_data/source_data/playground-series-s4e9/prepare.py
        :language: python
        :caption: ds_data/source_data/playground-series-s4e9/prepare.py
        :linenos:

      - At the end of program execution, the ``ds_data`` folder structure will look like this:

      .. code-block:: text

        ds_data
        ├── playground-series-s4e9
        │   ├── train.csv
        │   ├── test.csv
        │   └── sample_submission.csv
        ├── eval
        │   └── playground-series-s4e9
        │       └── submission_test.csv
        └── source_data
            └── playground-series-s4e9
                ├── prepare.py
                ├── sample_submission.csv
                ├── test.csv
                └── train.csv

    - Create a ``ds_data/playground-series-s4e9/description.md`` file to describe your competition, dataset description, and other information. We can find the `competition description information <https://www.kaggle.com/competitions/playground-series-s4e9/overview>`_ and the `dataset description information <https://www.kaggle.com/competitions/playground-series-s4e9/data>`_ from the Kaggle website.

      - The following shows the description file for ``playground-series-s4e9``

        .. literalinclude:: ../../rdagent/scenarios/data_science/example/ds_data/playground-series-s4e9/description.md
          :language: markdown
          :caption: ds_data/playground-series-s4e9/description.md
          :linenos:

    - Create a ``ds_data/eval/playground-series-s4e9/valid.py`` file, which is used to check the validity of the submission files to ensure that their formatting is consistent with the reference file.

      - The following shows a script that checks the validity of a submission based on the ``playground-series-s4e9`` data.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/ds_data/eval/playground-series-s4e9/valid.py
        :language: markdown
        :caption: ds_data/eval/playground-series-s4e9/valid.py
        :linenos:

    - Create a ``ds_data/eval/playground-series-s4e9/grade.py`` file, which is used to calculate the score based on the submission file and the **standard answer file**, and output the result in JSON format.

      - The following shows a grading script based on the ``playground-series-s4e9`` data implementation.

      .. literalinclude:: ../../rdagent/scenarios/data_science/example/ds_data/eval/playground-series-s4e9/grade.py
        :language: markdown
        :caption: ds_data/eval/playground-series-s4e9/grade.py
        :linenos:

  - In this example we don't create a ``ds_data/eval/playground-series-s4e9/sample.py``, we use the sample method provided by RD-Agent by default.

  - At this point, you have created a complete dataset. The correct structure of the dataset should look like this.

    .. code-block:: text

        ds_data
        ├── playground-series-s4e9
        │   ├── train.csv
        │   ├── test.csv
        │   ├── description.md
        │   └── sample_submission.csv
        ├── eval
        │   └── playground-series-s4e9
        │       ├── grade.py
        │       ├── submission_test.csv
        │       └── valid.py
        └── source_data
            └── playground-series-s4e9
                ├── prepare.py
                ├── sample_submission.csv
                ├── test.csv
                └── train.csv











- 📥 **Download Competition Data**

  - Kaggle competition data, contains two parts: competition description file (json file or markdown file) and competition dataset (zip file).

    - **How to get the competition dataset**

      - The competition dataset is downloaded and extracted automatically when the program is run. If the zip file exists, the download will be skipped, if the unzip folder exists, the unzip will be skipped.

    - **How to get the competition description file**

      - The competition description file is downloaded automatically when the programme is run, and the download process relies on ``chromedriver``, which can be installed as follows:

        .. code-block:: sh

          # install chrome
          wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
          sudo apt install ./google-chrome-stable_current_amd64.deb
          google-chrome --version
          # install chromedriver
          wget "https://storage.googleapis.com/chrome-for-testing-public/$(google-chrome --version | grep -oP '\d+\.\d+\.\d+\.\d+')/linux64/chromedriver-linux64.zip"
          unzip chromedriver-linux64.zip
          cd chromedriver-linux64
          sudo mv chromedriver /usr/local/bin
          sudo chmod +x /usr/local/bin/chromedriver
          chromedriver --version

    

    - **Correct directory structure (Here is an example of competition data with id sf-crime)**

      .. code-block:: text

        kaggle_data
        └── zip_files
        | └── sf-crime.zip
        ├── sample
        | └── sf-crime
        |   └── ...
        ├── sf-crime.json
        └── sf-crime
          └── ...
        
      - ``kaggle_data/zip_files/sf-crime.zip:`` Competition dataset zip files downloaded from the Kaggle website.

      - ``kaggle_data/sf-crime.json:`` Competition description file.

      - ``kaggle_data/sf-crime:`` The target folder for unzipping the competition dataset. Complete dataset.

      - ``kaggle_data/sample/sf-crime:`` Simplified dataset based on the complete dataset. Used to quickly verify that the code works.

- 🚀 **Run the Application**

  - You can directly run the application by using the following command:

    .. code-block:: sh

        rdagent kaggle --competition <Competition ID>

- 📤 **Submit the Result Automatically or Manually**

  - If Auto: You need to set ``KG_AUTO_SUBMIT`` to ``true`` in the ``.env`` file.

    .. code-block:: sh

      dotenv set KG_AUTO_SUBMIT true
  
  - Else: You can download the prediction results from the UI interface and submit them manually. For more details, refer to the :doc:`UI guide <../ui>`.




- **Kaggle Agent**, as a subset of **data science**, can also be run via the **data science command**, like this:

  .. code-block:: sh

      rdagent data_science --competition <your competition name>

  

🔍 MLE-bench Guide: Running ML Engineering via MLE-bench
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 📝 **MLE-bench Overview**

  - MLE-bench is a comprehensive benchmark designed to evaluate the ML engineering capabilities of AI systems using real-world scenarios. Since Kaggle does not provide held-out test sets for these competitions, the benchmark includes preparation scripts that split the publicly available training data into new training and test sets, and grading scripts are provided for each competition to accurately evaluate submission scores.

    - You can see which competitions are supported in the `repo of mle-bench <https://github.com/openai/mle-bench/tree/main/mlebench/competitions>`_.

- 🔧 **Set up Environment for MLE-bench**

  - Running R&D-Agent on MLE-bench is designed for full automation. There is no need for manual downloads and data preparation. Simply set the environment variable ``DS_IF_USING_MLE_DATA`` to True.  

  - At runtime, R&D-Agent will automatically build the Docker image specified at ``rdagent/scenarios/kaggle/docker/mle_bench_docker/Dockerfile``. This image is responsible for downloading the required datasets and grading files for MLE-bench.  
  
  - Note: The first run may take longer than subsequent runs as the Docker image and data are being downloaded and set up for the first time.

    .. code-block:: sh

        dotenv set DS_LOCAL_DATA_PATH <your local directory>/kaggle_data
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




