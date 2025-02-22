.. _model_copilot_general:

======================
General Model Copilot
======================

**🤖 Automated Model Research & Development Co-Pilot**
--------------------------------------------------------

📖 Background
~~~~~~~~~~~~~~
In the fast-paced field of artificial intelligence, the number of academic papers published each year is skyrocketing. 
These papers introduce new models, techniques, and approaches that can significantly advance the state of the art. 
However, reproducing and implementing these models can be a daunting task, requiring substantial time and expertise. 
Researchers often face challenges in extracting the essential details from these papers and converting them into functional code.
And this is where the **General Model Copilot** steps in.

🎥 `Demo <https://rdagent.azurewebsites.net/report_model>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div style="display: flex; justify-content: center; align-items: center;">
      <video width="600" controls>
        <source src="https://rdagent.azurewebsites.net/media/b35f904765b05099b0fcddbebe041a04f4d7bde239657e5fc24bf0cc.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>

🌟 Introduction
~~~~~~~~~~~~~~~~
In this scenario, our automated system proposes hypotheses, constructs models, implements code, performs back-testing, and uses feedback to iterate continuously. The system aims to automatically optimize performance metrics from the Qlib library, finding the best code through autonomous research and development.

Model R&D CoPilot Scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    
  - Prepare relevant files (in pdf format) by uploading papers to the directory below and copy the path as report_file_path.
      
    .. code-block:: sh

        rdagent/scenarios/general_model
    
  - Run the following command in your terminal within the same virtual environment:
  
    .. code-block:: sh

        rdagent general_model --report_file_path=<path_to_pdf_file>
