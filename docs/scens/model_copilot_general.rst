.. _model_copilot_general:

======================
General Model Copilot
======================

**🤖 Automated Model Research & Development Co-Pilot **

Scen5 Intro
~~~~~~~~~~~
In this scenario, our automated system proposes hypotheses, constructs models, implements code, performs back-testing, and uses feedback to iterate continuously. The system aims to automatically optimize performance metrics from the Qlib library, finding the best code through autonomous research and development.

Model R&D CoPilot Scenario
~~~~~~~~~~~~~~~~~~~~~~
**Overview**

This demo automates the extraction and iterative development of models from academic papers, ensuring functionality and correctness. This scenario automates the development of PyTorch models by reading academic papers or other sources. It supports various data types, including tabular, time-series, and graph data. The primary workflow involves two main components: the Reader and the Coder.

**Workflow Components**

1. **Reader**
   - Parses and extracts relevant model information from academic papers or sources, including architectures, parameters, and implementation details.
   - Uses Large Language Models to convert content into a structured format for the Coder.

2. **Evolving Coder**
   - Translates structured information from the Reader into executable PyTorch code.
   - Utilizes an evolving coding mechanism to ensure correct tensor shapes, verified with sample input tensors.
   - Iteratively refines the code to align with source material specifications.

**Supported Data Types**

- **Tabular Data:** Structured data with rows and columns, such as spreadsheets or databases.
- **Time-Series Data:** Sequential data points indexed in time order, useful for forecasting and temporal pattern recognition.
- **Graph Data:** Data structured as nodes and edges, suitable for network analysis and relational tasks.


Demo
~~~~~~~~~~
.. TODO

Quick Start
~~~~~~~~~~~~~~~~~
To quickly start the model co-pilot process, first prepare environment setup:

.. code-block:: sh

    cd rdagent
    make dev 

Then prepare relevant files (in pdf format) by uploading papers to the directory below and copy the path as report_file_path. 

.. code-block:: sh

    rdagent/scenarios/general_model

Run the following command in your terminal within the same virtual environment:

.. code-block:: sh

    python rdagent/app/general_model/general_model.py report_file_path 

Usage of modules
~~~~~~~~~~~~~~~~~
There are mainly two modules in this scenario: one that reads the paper and returns a model card & one that reads the model card and returns functional code. The moduldes can also be used separately as components for developers to build up new scenarios.


Configurations
~~~~~~~~~~~~~~~~~

The `config.yaml` file located in the `model_template` folder contains the relevant configurations for running the developed model in Qlib. The default settings include key information such as:

- **market**: Specifies the market, which is set to `csi300`.
- **fields_group**: Defines the fields group, with the value `feature`.
- **col_list**: A list of columns used, including various indicators such as `RESI5`, `WVMA5`, `RSQR5`, and others.
- **start_time**: The start date for the data, set to `2008-01-01`.
- **end_time**: The end date for the data, set to `2020-08-01`.
- **fit_start_time**: The start date for fitting the model, set to `2008-01-01`.
- **fit_end_time**: The end date for fitting the model, set to `2014-12-31`.

The default hyperparameters used in the configuration are as follows:

- **n_epochs**: The number of epochs, set to `100`.
- **lr**: The learning rate, set to `1e-3`.
- **early_stop**: The early stopping criterion, set to `10`.
- **batch_size**: The batch size, set to `2000`.
- **metric**: The evaluation metric, set to `loss`.
- **loss**: The loss function, set to `mse`.
- **n_jobs**: The number of parallel jobs, set to `20`.

