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