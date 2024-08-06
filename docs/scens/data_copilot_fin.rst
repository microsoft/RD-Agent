.. _data_copilot_fin:

=====================
Finance Data Copilot
=====================


**Automated Quantitative Trading & Factors Extraction from Financial Reports📄**
---------------------------------------------------------------------------------

Background
~~~~~~~~~~
**Research reports** are treasure troves of insights, often unveiling potential **factors** that can drive successful quantitative trading strategies. 
Yet, with the sheer volume of reports available, extracting the most valuable insights efficiently becomes a daunting task.

Furthermore, rather than hastily replicating factors from a report, it's essential to delve into the underlying logic of their construction. 
Does the factor capture the essential market dynamics? How unique is it compared to the factors already in your library?

Therefore, there is an urgent need for a systematic approach to design a framework that can effectively manage this process. 
This is where our R&D Agent comes into play.



Introduction
~~~~~~~~~~~~
In this scenario, our agent demonstrates the process of extracting factors from financial research reports, implementing these factors, and analyzing their performance through Qlib backtesting. 
This process continually expands and refines the factor library.

Here's an enhanced outline of the steps:

**Step 1 : Hypothesis Generation 🔍**

- Generate and propose initial hypotheses based on insights from financial reports with thorough reasoning and financial justification.

**Step 2 : Factor Creation ✨**

- Based on the hypothesis and financial reports, divide the tasks. 
- Each task involves developing, defining, and implementing a new financial factor, including its name, description, formulation, and variables.

**Step 3 : Factor Implementation 👨‍💻**

- Implement the factor code based on the description, evolving it as a developer would.
- Quantitatively validate the newly created factors.

**Step 4 : Backtesting with Qlib 📉**

- Integrate the full dataset into the factor implementation code and prepare the factor library.
- Perform backtesting using the Alpha158+ model in Qlib to assess the factor's effectiveness and performance.

+----------------+------------+----------------+----------------------------------------------------+
| Dataset        | Model      | Factors        | Data Split                                         |
+================+============+================+====================================================+
| CSI300         | LGBModel   | Alpha158 Plus  | +-----------+--------------------------+           |
|                |            |                | | Train     | 2008-01-01 to 2014-12-31 |           |
|                |            |                | +-----------+--------------------------+           |
|                |            |                | | Valid     | 2015-01-01 to 2016-12-31 |           |
|                |            |                | +-----------+--------------------------+           |
|                |            |                | | Test      | 2017-01-01 to 2020-08-01 |           |
|                |            |                | +-----------+--------------------------+           |
+----------------+------------+----------------+----------------------------------------------------+


**Step 5 : Feedback Analysis 🔍**
   - Analyze backtest results.
   - Incorporate feedback to refine hypotheses.

**Step 6 : Knowledge Base Refinement ♻️**
   - Refine the knowledge base based on feedback and repeat the process.

Demo
~~~~~~~~~~
.. TODO

Scen2 Quick Start
~~~~~~~~~~~~~~~~~

To quickly start the factor extraction process, run the following command in your terminal within the  `rdagent` virtual environment:

.. code-block:: sh

    python rdagent/app/qlib_rd_loop/factor_from_report_w_sc.py.py


Usage of modules
~~~~~~~~~~~~~~~~~
TODO: Show some examples:

