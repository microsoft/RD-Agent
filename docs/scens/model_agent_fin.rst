.. _model_agent_fin:

=======================
Finance Model Agent
=======================

**🤖 Automated Quantitative Trading & Iterative Model Evolution**
------------------------------------------------------------------------------------------

📖 Background
~~~~~~~~~~~~~~
TODO

🎥 Demo
~~~~~~~~~~
TODO: Here should put a video of the demo.


🌟 Introduction
~~~~~~~~~~~~~~~~

In this scenario, our automated system proposes hypothesis, constructs model, implements code, receives back-testing, and uses feedbacks. 
Hypothesis is iterated in this continuous process. 
The system aims to automatically optimise performance metrics from Qlib library thereby finding the optimised code through autonomous research and development.

Here's an enhanced outline of the steps:

**Step 1 : Hypothesis Generation 🔍**

- Generate and propose initial hypotheses based on previous experiment analysis and domain expertise, with thorough reasoning and financial justification.

**Step 2 : Model Creation ✨**

- Transform the hypothesis into a task.
- Develop, define, and implement a quantitative model, including its name, description, and formulation.

**Step 3 : Model Implementation 👨‍💻**

- Implement the model code based on the detailed description.
- Evolve the model iteratively as a developer would, ensuring accuracy and efficiency.

**Step 4 : Backtesting with Qlib 📉**

- Conduct backtesting using the newly developed model and 20 factors extracted from Alpha158 in Qlib.
- Evaluate the model's effectiveness and performance.

+----------------+------------+------------------------+----------------------------------------------------+
| Dataset        | Model      | Factors                | Data Split                                         |
+================+============+========================+====================================================+
| CSI300         | RDAgent-dev| 20 factors (Alpha158)  | +-----------+--------------------------+           |
|                |            |                        | | Train     | 2008-01-01 to 2014-12-31 |           |
|                |            |                        | +-----------+--------------------------+           |
|                |            |                        | | Valid     | 2015-01-01 to 2016-12-31 |           |
|                |            |                        | +-----------+--------------------------+           |
|                |            |                        | | Test      | 2017-01-01 to 2020-08-01 |           |
|                |            |                        | +-----------+--------------------------+           |
+----------------+------------+------------------------+----------------------------------------------------+

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

- 📦 Install Pytorch
    - Install Pytorch and related libraries:

      .. code-block:: sh

          pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
          pip3 install torch_geometric

- ⚙️ Environment Configuration
    - Place the `.env` file in the same directory as the `.env.example` file.
        - The `.env.example` file contains the environment variables required for users using the OpenAI API (Please note that `.env.example` is an example file. `.env` is the one that will be finally used.)

    - Export each variable in the .env file:

      .. code-block:: sh

          export $(grep -v '^#' .env | xargs)

- 🚀 Run the Application
    .. code-block:: sh

        python rdagent/app/qlib_rd_loop/model_w_sc.py

🛠️ Usage of modules
~~~~~~~~~~~~~~~~~~~~~
TODO: Show some examples:
