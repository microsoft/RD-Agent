# Detailed Explanation for Customized Data in R\&D-Agent Data Science Pipeline

R\&D-Agent Data Science Pipeline supports automated R\&D optimization for competitions hosted on the Kaggle platform, as well as **custom user-defined datasets**.

Specifically, you need to prepare files in a structure similar to the provided example. Here, we use the `arf-12-hour-prediction-task` dataset as an illustration.

## arf-12-hour-prediction-task Introduction

> Acute Respiratory Failure (ARF) is a life-threatening condition that often develops rapidly in critically ill patients. Accurate early prediction of ARF is essential in Intensive Care Units (ICUs) to enable timely clinical interventions and effective resource allocation. In this task, you are required to build a machine learning model that predicts whether a patient will develop ARF within the next **12 hours**, using multivariate clinical time-series data.
> 
> The dataset has been extracted from electronic health records (EHRs) and preprocessed through the **FIDDLE** pipeline, generating structured temporal features for each patient.

## Example Folder Structure

* `arf-12-hour-prediction-task` (Task Name)

  * `train` (**required**)

    * `X.npz` (features)
    * `ARF_12h.csv` (labels)
  * `test` (**required**)

    * `X.npz` (features)
    * `ARF_12h.csv` (labels to be predicted)
  * `description.md` (**required**): A detailed description of the task, including sections such as Task Description, Objective, Data Description, Modeling, and Submission Format.
  * `sample.py` (**optional**): A Python script to sample the dataset for debugging purposes. If not provided, a default sampling logic in R\&D-Agent will be used. Refer to the `create_debug_data` function in `rdagent/scenarios/data_science/debug/data.py`.

* `eval` (**required**)

  * `arf-12-hour-prediction-task` (Task Name, **required**)

    * `grade.py`: Calculates the task score on the test dataset.
    * `submission_test.csv`: Corresponding labels from the previously provided `test/ARF_12h.csv`.
    * `valid.py`: Checks the validity of the generated `submission.csv` file.

The complete dataset folder for `arf-12-hour-prediction-task` can be downloaded from:
[https://github.com/SunsetWolf/rdagent\_resource/releases/download/med\_data/rdagent\_datas\_science\_customData\_example.zip](https://github.com/SunsetWolf/rdagent_resource/releases/download/med_data/rdagent_datas_science_customData_example.zip).

The original dataset is sourced from PhysioNet. You can apply for an account at [PhysioNet](https://physionet.org/) and then request access to the FIDDLE preprocessed data: [FIDDLE Dataset](https://physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/).
