.. _data_agent_fin:

=====================
Finance Data Agent
=====================


**Automated Quantitative Trading & Iterative Factors Evolution 🤖**
-------------------------------------------------------------------

Background
~~~~~~~~~~
In the dynamic world of quantitative trading, **factors** are the secret weapons that traders use to harness market inefficiencies. 

These powerful tools—ranging from straightforward metrics like price-to-earnings ratios to intricate discounted cash flow models—unlock the potential to predict stock prices with remarkable precision. 
By tapping into this rich vein of data, quantitative traders craft sophisticated strategies that not only capitalize on market patterns but also drastically enhance trading efficiency and accuracy. 

Embrace the power of factors, and you're not just trading; you're strategically outsmarting the market.


Introduction
~~~~~~~~~~~~
In this scenario, our agent illustrates the iterative process of hypothesis generation, knowledge construction, and decision-making. 

It highlights how financial factors evolve through continuous feedback and refinement. 

Here's an enhanced outline of the steps:

**Step 1 : Hypothesis Generation 🔍**

- Generate and propose initial hypotheses based on previous experiment analysis and domain expertise, with thorough reasoning and financial justification.

**Step 2 : Factor Creation ✨**

- Based on the hypothesis, divide the tasks.
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

**Step 6 :Hypothesis Refinement ♻️**
   - Refine hypotheses based on feedback and repeat the process.

Demo
~~~~~~~~~~
.. TODO

Quick Start
~~~~~~~~~~~~~~~~~

To quickly start the Automated Quantitative Trading & Iterative Factors Evolution process, run the following command in your terminal within the `rdagent` virtual environment:

.. code-block:: sh

    python rdagent/app/qlib_rd_loop/factor_w_sc.py


Usage of modules
~~~~~~~~~~~~~~~~~
TODO: Show some examples:

