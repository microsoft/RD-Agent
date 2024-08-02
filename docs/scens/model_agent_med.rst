.. _model_agent_med:

===================
Medical Model Agent
===================


Scen4: 
------
🤖 Automated Model Research & Development Co-Pilot 

Scen4 Intro
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
To quickly start the model proposal process, run the following command in your terminal within the 'rdagent' virtual environment:

.. code-block:: sh

    python rdagent/app/model_extraction_and_code/model_extraction_and_implementation.py

Usage of modules
~~~~~~~~~~~~~~~~~
TODO: Show some examples

