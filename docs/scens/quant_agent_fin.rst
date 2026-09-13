.. _quant_agent_fin:

=====================
Finance Quant Agent
=====================


**🥇The First Data-Centric Quant Multi-Agent Framework RD-Agent(Q)**
---------------------------------------------------------------------

R&D-Agent for Quantitative Finance, in short **RD-Agent(Q)**, is the first data-centric, multi-agent framework designed to automate the full-stack research and development of quantitative strategies via coordinated factor-model co-optimization.

You can learn more details about **RD-Agent(Q)** through the `paper <https://arxiv.org/abs/2505.15155>`_.

⚡ Quick Start
~~~~~~~~~~~~~~~~~

Before you start, please make sure you have installed RD-Agent and configured the environment for RD-Agent correctly. If you want to know how to install and configure the RD-Agent, please refer to the `documentation <../installation_and_configuration.html>`_.

Also prepare the :ref:`qlib-data-prerequisites` before running ``rdagent fin_quant``.

Then, you can run the framework by running the following command:

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


.. _qlib-data-prerequisites:

Qlib data prerequisites
~~~~~~~~~~~~~~~~~~~~~~~

The default ``fin_quant``, ``fin_factor``, and ``fin_model`` scenarios use
**daily Chinese-market data in Qlib's binary format**. Installing RD-Agent does
not install this dataset. Follow the Linux setup above and prepare data under
the same host user that will run RD-Agent.

1. Obtain the dataset using Qlib's `Data Preparation guide
   <https://github.com/microsoft/qlib#data-preparation>`_. Qlib currently notes
   that its official dataset is temporarily unavailable and links to a
   community-maintained alternative. Check that guide for the current source
   and download instructions; do not assume the automatic download will succeed.

2. Populate ``~/.qlib/qlib_data/cn_data`` on the host. For the community
   ``qlib_bin.tar.gz`` archive layout shown in Qlib's guide, extract the downloaded
   archive as follows:

   .. code-block:: sh

       mkdir -p "$HOME/.qlib/qlib_data/cn_data"
       tar -xzf /path/to/qlib_bin.tar.gz \
           -C "$HOME/.qlib/qlib_data/cn_data" --strip-components=1

   Replace the archive path with your downloaded file. The resulting directory
   must directly contain ``calendars/``, ``instruments/``, and ``features/``,
   rather than another nested ``qlib_bin/`` directory. Use daily data with the
   CSI 300 instrument list and dates covering the selected Qlib YAML template.

3. Check the layout before starting a finance command:

   .. code-block:: sh

       python - "$HOME/.qlib/qlib_data/cn_data" <<'PY'
       import sys
       from pathlib import Path

       root = Path(sys.argv[1])
       for relative in ("calendars/day.txt", "instruments/csi300.txt"):
           path = root / relative
           if not path.is_file() or path.stat().st_size == 0:
               raise SystemExit(f"Missing or empty Qlib data file: {path}")
       if not any((root / "features").glob("*/*.day.bin")):
           raise SystemExit(f"No daily Qlib feature files under {root / 'features'}")
       print(f"Qlib data layout found at {root}")
       PY

   This checks the expected layout, not data quality or date coverage. Qlib's
   guide also documents its data-health checks.

With the default ``QlibDockerConf``, the host's ``~/.qlib`` is mounted at
``/root/.qlib`` inside the container. The template ``provider_uri`` of
``~/.qlib/qlib_data/cn_data`` therefore refers to that mounted dataset during a
Docker backtest. Downloading data only into an unrelated container does not
populate the host path. If you customize the mount or ``provider_uri``, ensure
they still refer to the same dataset.

``QTDockerEnv.prepare()`` attempts to download data only when ``cn_data`` does
not exist. An empty or partially downloaded directory can therefore cause
"Data already exists. Download skipped." without usable data. Inspect and
repair that dataset using the steps above instead of repeatedly restarting the
agent.

Factor preparation derives ``daily_pv.h5`` from the Qlib dataset in each of
these default directories, relative to the RD-Agent working directory:

* ``git_ignore_folder/factor_implementation_source_data`` (full data)
* ``git_ignore_folder/factor_implementation_source_data_debug`` (debug data)

These are derived factor inputs, not replacements for the binary Qlib dataset.
Their locations can be set with ``FACTOR_CoSTEER_data_folder`` and
``FACTOR_CoSTEER_data_folder_debug``. If ``daily_pv_all.h5 is not generated`` or
``daily_pv_debug.h5 is not generated`` appears, inspect the preceding container
log and verify the source dataset and mount first. Merely creating the derived
directories does not generate their contents.


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

- **Qlib Configuration**
    - The `.yaml` files in both the `model_template` and `factor_template` directories contain some configurations for running the corresponding models or factors within the Qlib framework. Below is an overview of their contents and roles:
        - **General Settings**:
            - **provider_uri**: Specifies the local Qlib data path, set to `~/.qlib/qlib_data/cn_data`.
            - **market**: Configured to `csi300`, representing the CSI 300 index constituents.
            - **benchmark**: Set to `SH000300`, used for backtesting evaluation.
        
        - **Data Handling**:
            - **start_time** and **end_time**: Define the full data range, from `2008-01-01` to `2022-08-01`.
            - **fit_start_time**: The start date for fitting the model, set to `2008-01-01`.
            - **fit_end_time**: The end date for fitting the model, set to `2014-12-31`.
            - **features and labels**: Generated via a nested data loader combining `Alpha158DL` (for engineered features such as `RESI5`, `WVMA5`, `RSQR5`, `KLEN`, etc.) and a `StaticDataLoader` that loads precomputed factor files (`combined_factors_df.parquet`).
            -  **normalization**: The pipeline includes `RobustZScoreNorm` (with clipping) and `Fillna` for inference, and `DropnaLabel` with `CSZScoreNorm` for training.
        
        - **Training Configuration**:
            - **Model**: Uses `GeneralPTNN`, a PyTorch-based neural network model.
            - **Dataset Splits**:
                - **train**: `2008-01-01` to `2014-12-31`
                - **valid**: `2015-01-01` to `2016-12-31`
                - **test**: `2017-01-01` to `2020-08-01`

        - **Default Hyperparameters** (can be overridden by command-line arguments):
            - **n_epochs**: `100`
            - **lr**: `2e-4`
            - **early_stop**: `10`
            - **batch_size**: `256`
            - **weight_decay**: `0.0`
            - **metric**: `loss`
            - **loss**: `mse`
            - **n_jobs**: `20`
            - **GPU**: `0` (uses GPU 0 if available)
            
        - **Backtesting and Evaluation**:
            - **strategy**: `TopkDropoutStrategy`, which selects the top 50 stocks and randomly drops 5 to introduce exploration.
            - **backtest period**: `2017-01-01` to `2020-08-01`
            - **initial capital**: `100,000,000`
            - **cost configuration**: Includes open/close costs, minimum transaction costs, and slippage control.
            
        - **Recording and Analysis**:
            - **SignalRecord**: Logs predicted signals.
            - **SigAnaRecord**: Performs signal analysis without long-short separation.
            - **PortAnaRecord**: Conducts portfolio analysis using the configured strategy and backtest settings.
