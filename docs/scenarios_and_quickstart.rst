=========================
Scenarios and Quick Start
=========================

Scenario lists
=========================

.. list-table:: 
   :header-rows: 1

   * - Scenario/Target
     - Model Implementation
     - Data Building
   * - 💹 Finance
     - Iteratively Proposing Ideas & Evolving
     - Auto reports reading & implementation
       Iteratively Proposing Ideas & Evolving
   * - 🩺 Medical
     - Iteratively Proposing Ideas & Evolving
     - 
   * - 🏭 General
     - Auto paper reading & implementation
     - 

Scnarios' demo & quick start
============================

Scen1
-----
🤖 Knowledge-Based Hypothesis Generation and Iteration

Scen1 Intro
~~~~~~~~~~~
In this scenario, our model autonomously generates and tests hypotheses using a knowledge base. The process involves:

- **🔍 Hypothesis Generation**: The model proposes new hypotheses.
- **📝 Factor Creation**: Write and define new factors.
- **✅ Factor Validation**: Validate the factors quantitatively.
- **📈 Backtesting with Qlib**: 

  - **Dataset**: CSI300
  - **Model**: LGBModel
  - **Factors**: Alpha158 +
  - **Data Split**:

    - **Train**: 2008-01-01 to 2014-12-31
    - **Valid**: 2015-01-01 to 2016-12-31
    - **Test**: 2017-01-01 to 2020-08-01
- **🔄 Feedback Analysis**: Analyze backtest results.
- **🔧 Hypothesis Refinement**: Refine hypotheses based on feedback and repeat.

Scen1 Demo
~~~~~~~~~~
.. TODO

Scen1 Quick Start
~~~~~~~~~~~~~~~~~

To quickly start the factor extraction process, run the following command in your terminal within the 'rdagent' virtual environment:

.. code-block:: sh

    python rdagent/app/qlib_rd_loop/factor.py


Usage of modules
~~~~~~~~~~~~~~~~~
TODO: Show some examples:


Scen2: 
------
📄 Research Report-Based Factor Extraction

Scen2 Intro
~~~~~~~~~~~
In this scenario, factors and hypotheses are extracted from research reports. The process includes:

- **🔍 Factor Extraction**: Extract relevant factors from research reports.
- **📝 Factor Creation**: Define these extracted factors.
- **✅ Factor Validation**: Validate the extracted factors.
- **📈 Backtesting with Qlib**: 

  - **Dataset**: CSI300
  - **Model**: LGBModel
  - **Factors**: Alpha158 +
  - **Data Split**:

    - **Train**: 2008-01-01 to 2014-12-31
    - **Valid**: 2015-01-01 to 2016-12-31
    - **Test**: 2017-01-01 to 2020-08-01
- **🔄 Feedback Analysis**: Analyze backtest results.
- **🔧 Hypothesis Refinement**: Refine hypotheses based on feedback and continue the cycle.

Scen2 Demo
~~~~~~~~~~
.. TODO

Scen2 Quick Start
~~~~~~~~~~~~~~~~~

To quickly start the factor extraction process, run the following command in your terminal within the 'rdagent' virtual environment:

.. code-block:: sh

    python rdagent/app/qlib_rd_loop/factor_from_report_sh.py


Usage of modules
~~~~~~~~~~~~~~~~~
TODO: Show some examples:

Scen3: 
------
🤖 Automated Machine Learning Model Construction & Iteration in Quantitative Finance 
Scen3 Intro
~~~~~~~~~~~
In this scenario, our automated system proposes hypothesis, constructs model, implements code, receives back-testing, and uses feedbacks. Hypothesis is iterated in this continuous process. The system aims to automatically optimise performance metrics from Qlib library thereby finding the optimised code through autonomous research and development.

Automated R&D of model in Quantitative Finance
~~~~~~~~~~~~~
**R (Research)**
- Iteration of ideas and hypotheses.
- Continuous learning and knowledge construction.

**D (Development)**
- Evolving code generation and model refinement.
- Automated implementation and testing of models.

Objective
~~~~~~~~~
The demo showcases the iterative process of hypothesis generation, knowledge construction, and decision-making in model construction in quantitative finance. It highlights how models evolve through continuous feedback and refinement, therefore building the domain knowledge specific to the scenario of quantitative finance.

Scen3 Demo
~~~~~~~~~~
.. TODO
Scen3 Quick Start
~~~~~~~~~~~~~~~~~

To quickly start the model proposal process, run the following command in your terminal within the 'rdagent' virtual environment:

.. code-block:: sh

    python rdagent/app/qlib_rd_loop/model_w_sc.py

Usage of modules
~~~~~~~~~~~~~~~~~
TODO: Show some examples:

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

