.. _model_agent_med:

=======================
Medical Model Agent
=======================

**🤖 Automated Medical Predtion Model Evolution**
------------------------------------------------------------------------------------------

📖 Background
~~~~~~~~~~~~~~
In this scenario, we consider the problem of risk prediction from patients' ICU monitoring data. We use the a public EHR dataset - MIMIC-III and extract a binary classification task for evaluating the framework.
In this task, we aim at predicting the whether the patients will suffer from Acute Respiratory Failure (ARF) based their first 12 hours ICU monitoring data. 

🎥 Demo
~~~~~~~~~~
TODO: Here should put a video of the demo.


🌟 Introduction
~~~~~~~~~~~~~~~~

In this scenario, our automated system proposes hypothesis, constructs model, implements code, receives back-testing, and uses feedbacks. 
Hypothesis is iterated in this continuous process. 
The system aims to automatically optimise performance metrics of medical prediction thereby finding the optimised code through autonomous research and development.

Here's an enhanced outline of the steps:

**Step 1 : Hypothesis Generation 🔍**

- Generate and propose initial hypotheses based on previous experiment analysis and domain expertise, with thorough reasoning and justification.

**Step 2 : Model Creation ✨**

- Transform the hypothesis into a model.
- Develop, define, and implement a machine learning model, including its name, description, and formulation.

**Step 3 : Model Implementation 👨‍💻**

- Implement the model code based on the detailed description.
- Evolve the model iteratively as a developer would, ensuring accuracy and efficiency.

**Step 4 : Backtesting with MIMIC-III 📉**

- Conduct backtesting using the newly developed model on the extracted task from MIMIC-III.
- Evaluate the model's effectiveness and performance in terms of AUROC score.

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

- 📦 Install the RDAgent
    - You can directly install the RDAgent package from PyPI:

      .. code-block:: sh

          pip install rdagent

- 📦 Request PhysioNet Account
    - Apply for an account at `PhysioNet <https://physionet.org/>`_.
    - Request access to FIDDLE preprocessed data: `FIDDLE Dataset <https://physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/>`_.
    - Place your username and password in `.rdagent.app.data_mining.conf`.

- ⚙️ Environment Configuration
    - Place the `.env` file in the same directory as the `.env.example` file.
        - The `.env.example` file contains the environment variables required for users using the OpenAI API (Please note that `.env.example` is an example file. `.env` is the one that will be finally used.)

    - Export each variable in the .env file:

      .. code-block:: sh

          export $(grep -v '^#' .env | xargs)
    
    - If you want to change the default environment variables, you can refer to `Env Config`_ below

- 🚀 Run the Application
    .. code-block:: sh

        rdagent med_model

🛠️ Usage of modules
~~~~~~~~~~~~~~~~~~~~~

.. _Env Config: 

- **Env Config**

The following environment variables can be set in the `.env` file to customize the application's behavior:

.. autopydantic_settings:: rdagent.app.data_mining.conf.PropSetting
    :settings-show-field-summary: False
    :exclude-members: Config