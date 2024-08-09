.. _data_agent_fin:

=====================
Finance Data Agent
=====================


**🤖 Automated Quantitative Trading & Iterative Factors Evolution**
-------------------------------------------------------------------

📖 Background
~~~~~~~~~~~~~~
In the dynamic world of quantitative trading, **factors** are the secret weapons that traders use to harness market inefficiencies. 

These powerful tools—ranging from straightforward metrics like price-to-earnings ratios to intricate discounted cash flow models—unlock the potential to predict stock prices with remarkable precision. 
By tapping into this rich vein of data, quantitative traders craft sophisticated strategies that not only capitalize on market patterns but also drastically enhance trading efficiency and accuracy. 

Embrace the power of factors, and you're not just trading; you're strategically outsmarting the market.


🎥 Demo
~~~~~~~~~~
TODO: Here should put a video of the demo.


🌟 Introduction
~~~~~~~~~~~~~~~~
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
- Conduct backtesting using the Alpha158 plus newly developed factors and LGBModel in Qlib to evaluate the new factors' effectiveness and performance.

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

- Analyze backtest results to assess performance.
- Incorporate feedback to refine hypotheses and improve the model.

**Step 6 :Hypothesis Refinement ♻️**

- Refine hypotheses based on feedback from backtesting.
- Repeat the process to continuously improve the model.

⚡ Quick Start
~~~~~~~~~~~~~~~~~

You can try our demo by running the following command:

- 🐍 Create a Conda Environment
    - Create a new conda environment with Python (3.10 and 3.11 are well tested in our CI):
    
      .. code-block:: sh
      
          conda create -n rdagent python=3.10

    - Activate the environment:

      .. code-block:: sh

          conda activate rdagent

- 🛠️ Run Make Files
    - Navigate to the directory containing the MakeFile and set up the development environment:

      .. code-block:: sh

          make dev

- ⚙️ Environment Configuration
    - Place the `.env` file in the same directory as the `.env.example` file.
        - The `.env.example` file contains the environment variables required for users using the OpenAI API (Please note that `.env.example` is an example file. `.env` is the one that will be finally used.)
    
    - If you want to change the default environment variables, you can refer to `Env Config`_ below

- 🚀 Run the Application
    .. code-block:: sh

        rdagent fin_factor


🛠️ Usage of modules
~~~~~~~~~~~~~~~~~~~~~

.. _Env Config: 

- **Env Config**

The following environment variables can be set in the `.env` file to customize the application's behavior:
    - **Path to the folder containing private data (default fundamental data in Qlib):**

        .. code-block:: sh

          FACTOR_CODER_DATA_FOLDER=/path/to/data/factor_implementation_source_data_all

    - **Path to the folder containing partial private data (for debugging):**

      .. code-block:: sh

          FACTOR_CODER_DATA_FOLDER_DEBUG=/path/to/data/factor_implementation_source_data_debug

    - **Maximum time (in seconds) for writing factor code:**

      .. code-block:: sh

          FACTOR_CODER_FILE_BASED_EXECUTION_TIMEOUT=300

    - **Maximum number of factors to write in one experiment:**

      .. code-block:: sh

          FACTOR_CODER_SELECT_THRESHOLD=5

    - **Number of developing loops for writing factors:**

      .. code-block:: sh

          FACTOR_CODER_MAX_LOOP=10

