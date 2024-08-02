.. _data_agent_fin:

=====================
Finance Data Agent
=====================

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

