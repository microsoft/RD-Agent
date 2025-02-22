.. _model_agent_fin:

=======================
Finance Model Agent
=======================

**🤖 Automated Quantitative Trading & Iterative Model Evolution**
------------------------------------------------------------------------------------------

📖 Background
~~~~~~~~~~~~~~
In the realm of quantitative finance, both factor discovery and model development play crucial roles in driving performance. 
While much attention is often given to the discovery of new financial factors, the **models** that leverage these factors are equally important. 
The effectiveness of a quantitative strategy depends not only on the factors used but also on how well these factors are integrated into robust, predictive models.

However, the process of developing and optimizing these models can be labor-intensive and complex, requiring continuous refinement and adaptation to ever-changing market conditions. 
And this is where the **Finance Model Agent** steps in.


🎥 `Demo <https://rdagent.azurewebsites.net/model_loop>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div style="display: flex; justify-content: center; align-items: center;">
      <video width="600" controls>
        <source src="https://rdagent.azurewebsites.net/media/d85e8cab1da1cd3501d69ce837452f53a971a24911eae7bfa9237137.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>


🌟 Introduction
~~~~~~~~~~~~~~~~

In this scenario, our automated system proposes hypothesis, constructs model, implements code, conducts back-testing, and utilizes feedback in a continuous, iterative process.

The goal is to automatically optimize performance metrics within the Qlib library, ultimately discovering the most efficient code through autonomous research and development.

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
    
  - You can directly run the application by using the following command:
    
    .. code-block:: sh

        rdagent fin_model

🛠️ Usage of modules
~~~~~~~~~~~~~~~~~~~~~

.. _Env Config: 

- **Env Config**

The following environment variables can be set in the `.env` file to customize the application's behavior:

.. autopydantic_settings:: rdagent.app.qlib_rd_loop.conf.ModelBasePropSetting
    :settings-show-field-summary: False
    :exclude-members: Config

- **Qlib Config**
    - The `config.yaml` file located in the `model_template` folder contains the relevant configurations for running the developed model in Qlib. The default settings include key information such as:
        - **market**: Specifies the market, which is set to `csi300`.
        - **fields_group**: Defines the fields group, with the value `feature`.
        - **col_list**: A list of columns used, including various indicators such as `RESI5`, `WVMA5`, `RSQR5`, and others.
        - **start_time**: The start date for the data, set to `2008-01-01`.
        - **end_time**: The end date for the data, set to `2020-08-01`.
        - **fit_start_time**: The start date for fitting the model, set to `2008-01-01`.
        - **fit_end_time**: The end date for fitting the model, set to `2014-12-31`.

    - The default hyperparameters used in the configuration are as follows:
        - **n_epochs**: The number of epochs, set to `100`.
        - **lr**: The learning rate, set to `1e-3`.
        - **early_stop**: The early stopping criterion, set to `10`.
        - **batch_size**: The batch size, set to `2000`.
        - **metric**: The evaluation metric, set to `loss`.
        - **loss**: The loss function, set to `mse`.
        - **n_jobs**: The number of parallel jobs, set to `20`.
