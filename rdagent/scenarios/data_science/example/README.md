# Detailed Explanation for Customized Data in R\&D-Agent Data Science Pipeline

R\&D-Agent Data Science Pipeline supports automated R\&D optimization for competitions hosted on the Kaggle platform, as well as **custom user-defined datasets**.

Specifically, you need to prepare files in a structure similar to the provided example. Here, we use the `arf-12-hours-prediction-task` dataset as an illustration.

## arf-12-hours-prediction-task Introduction

> Acute Respiratory Failure (ARF) is a life-threatening condition that often develops rapidly in critically ill patients. Accurate early prediction of ARF is essential in Intensive Care Units (ICUs) to enable timely clinical interventions and effective resource allocation. In this task, you are required to build a machine learning model that predicts whether a patient will develop ARF within the next **12 hours**, using multivariate clinical time-series data.
> 
> The dataset has been extracted from electronic health records (EHRs) and preprocessed through the **FIDDLE** pipeline, generating structured temporal features for each patient.

## Example Folder Structure

* `source_data` (**required**)

  * `arf-12-hours-prediction-task` (Task Name, **required**)

    * `prepare.py` Used for data preprocessing to split the raw data into: *training data*, *test data*, *formatted submission file*, and *standard answer file*. 

  * `playground-series-s4e9` (Task Name, **required**)

    * `prepare.py` (**required**): Used for data preprocessing to split the raw data into: *training data*, *test data*, *formatted submission file*, and *standard answer file*. 

  NOTE: Due to the large size of the raw data, we do not show the raw data in this project, if you want to see the raw data, you can download the full dataset through the link at the bottom.

* `arf-12-hours-prediction-task` (Task Name)

  * `description.md` (**required**): A detailed description of the task, including sections such as *Task Description*, *Objective*, *Data Description*, *Data usage Notes*, *Modeling*, *Evaluation* and *Submission Format*.

  * `sample.py` (**optional**): A Python script to sample the dataset for debugging purposes. If not provided, a default sampling logic in R\&D-Agent will be used. Refer to the `create_debug_data` function in `rdagent/scenarios/data_science/debug/data.py`.

* `playground-series-s4e9` (Task Name)

  * `description.md` (**required**): A detailed description of the task, including sections such as *Task Description*, *Goal*, *Evaluation*, *Data Description*, and *Submission Format*.

* `eval` (**optional**)

  * `arf-12-hours-prediction-task` (Task Name, **optional**)

    * `grade.py`: Calculates the task score on the test dataset.
    * `valid.py`: Checks the validity of the generated `submission.csv` file.

  * `playground-series-s4e9` (Task Name, **optional**)

    * `grade.py`: Calculates the task score on the test dataset.
    * `valid.py`: Checks the validity of the generated `submission.csv` file.

  NOTE: You don't need to create the `eval` folder if you are ignoring test set scores.

---

The complete dataset folder for `arf-12-hours-prediction-task` can be downloaded from [here](https://github.com/SunsetWolf/rdagent_resource/releases/download/ds_data/arf-12-hours-prediction-task.zip).

The raw dataset for `arf-12-hours-prediction-task` comes from PhysioNet. You can apply for an account at [PhysioNet](https://physionet.org/) and then request access to the FIDDLE preprocessed data: [FIDDLE Dataset](https://physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/).

---

The complete dataset folder for `playground-series-s4e9` can be downloaded from [here](https://github.com/SunsetWolf/rdagent_resource/releases/download/ds_data/playground-series-s4e9.zip).

The raw dataset for `playground-series-s4e9` comes from Kaggle. You can apply for an account at [Kaggle](https://www.kaggle.com/) and then request access to the [competition dataset](https://www.kaggle.com/competitions/playground-series-s4e9/data).

---

**NOTE:** For more information about the dataset, please refer to the [documentation](https://rdagent.readthedocs.io/en/latest/scens/data_science.html).
