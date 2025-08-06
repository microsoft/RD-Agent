.. _finetune_agent:

=============================
Fine-tuning an Existing Model
=============================

## **🎯 Scenario: Continue Training on a Pre-trained Model**

In this workflow the **Data Science Agent** starts from a *previously trained* model (and its training script), performs additional fine-tuning on new data, and then re-uses the updated weights for subsequent inference runs.

🚧 Directory Structure

Your competition folder (here called ``custom_data``) must contain **one extra sub-directory** named ``prev_model`` where you keep the old weights and the code that produced them:

.. code-block:: text

   ds_data
   └── custom_data
       ├── train.csv
       ├── test.csv
       ├── sample_submission.csv      # optional
       ├── description.md             # optional
       ├── sample.py                  # optional
       └── prev_model                 # ← NEW
           ├── models/                #   previous checkpoints (e.g. *.bin, *.pt, *.ckpt)
           └── main.py                  #   training/inference scripts you used before

If your competition provides custom grading/validation scripts, keep them under ``ds_data/eval/custom_data`` exactly as before.

🔧 Environment Setup
~~~~~~~~~~~~~~~~~~~~~~

Add or update the following variables in **.env** (examples shown):

.. code-block:: sh

   # required for all Data-Science runs
   dotenv set DS_LOCAL_DATA_PATH <your local path>/ds_data

   # optional: choose docker / conda, etc.
   dotenv set DS_CODER_COSTEER_ENV_TYPE docker

🚀 How It Works at Runtime

1. **First run**

   * `rdagent` detects `prev_model/models`.
   * It loads the latest checkpoint and prepare the fine-tuning based on code found under `prev_model/*.py` (or your own pipeline if you override it).
   * Fine-tuned weights are written to `./workspace_input/models`.

2. **Subsequent runs**

   * When you execute `python ./workspace_input/main.py`, the script first looks for a checkpoint in `./workspace_input/models`.
   * If found, it **skips fine-tuning** and goes straight to prediction / submission generation.

⏰ Managing Timeouts


By default:

* **Debug loop**: 1 hour (``DS_DEBUG_TIMEOUT=3600`` seconds)  
* **Full run**  : 3 hours (``DS_FULL_TIMEOUT=10800`` seconds)

Override either value in **.env**:

.. code-block:: sh

   # give the debug loop 45 min and the full loop 6 h
   dotenv set DS_DEBUG_TIMEOUT 2700
   dotenv set DS_FULL_TIMEOUT 21600

- 🚀 **Run the Application**

  - You can directly run the application by using the following command:
    
    .. code-block:: sh

        dotenv run -- python rdagent/app/finetune/data_science/loop.py --competition <Competition ID>

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

- **Additional Guidance**

  - **Combine different LLM Models at R&D Stage**

    - You can combine different LLM models at the R&D stage. 

    - By default, when you set environment variable ``CHAT_MODEL``, it covers both R&D stages. When customizing the model for the development stage, you can set:
    
    .. code-block:: sh

      # This example sets the model to "o3-mini". For some models, the reasoning effort shoule be set to "None".
      dotenv set LITELLM_CHAT_MODEL_MAP '{"coding":{"model":"o3-mini","reasoning_effort":"high"},"running":{"model":"o3-mini","reasoning_effort":"high"}}'

