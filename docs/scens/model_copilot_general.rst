.. _model_copilot_general:

======================
General Model Copilot
======================


Scen5: 
------
🤖 Automated Model Research & Development Co-Pilot 

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

#### Supported Data Types

- **Tabular Data:** Structured data with rows and columns, such as spreadsheets or databases.
- **Time-Series Data:** Sequential data points indexed in time order, useful for forecasting and temporal pattern recognition.
- **Graph Data:** Data structured as nodes and edges, suitable for network analysis and relational tasks.

Scen4 Demo
~~~~~~~~~~
.. TODO

Scen4 Quick Start
~~~~~~~~~~~~~~~~~
To quickly start the model co-pilot process, first prepare environment setup:

.. code-block:: sh

    cd rdagent
    make dev 

Then prepare relevant files (in pdf format) by uploading papers to the directory below and copy the path as report_file_path. 

.. code-block:: sh

    /home/v-xisenwang/RD-Agent/rdagent/scenarios/general_model

Run the following command in your terminal within the same virtual environment:

.. code-block:: sh

    python rdagent/app/general_model/general_model.py report_file_path 

Usage of modules
~~~~~~~~~~~~~~~~~
There are mainly two modules in this scenario: one that reads the paper and returns a model card & one that reads the model card and returns functional code. The moduldes can also be used separately as components for developers to build up new scenarios.

