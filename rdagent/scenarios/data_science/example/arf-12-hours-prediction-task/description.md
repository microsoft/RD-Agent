# Competition name: ARF 12-Hour Prediction Task

## Overview

### Description

Acute Respiratory Failure (ARF) is a life-threatening condition that often develops rapidly in critically ill patients. Accurate early prediction of ARF is crucial in intensive care units (ICUs) to enable timely clinical interventions and resource allocation. In this task, you are asked to build a machine learning model that predicts whether a patient will develop ARF within the next **12 hours**, based on multivariate clinical time series data.

The dataset is extracted from electronic health records (EHRs) and preprocessed using the **FIDDLE** pipeline to generate structured temporal features for each patient.

### Objective

**Your Goal** is to develop a binary classification model that takes a 12-hour time series as input and predicts whether ARF will occur (1) or not (0) in the following 12 hours.

---

## Data Description

1. train/ARF_12h.csv: A CSV file containing the ICU stay ID, the hour of ARF onset, and the binary label indicating whether ARF will occur in the next 12 hours.

    * Columns: ID, ARF_ONSET_HOUR, ARF_LABEL

2. train/X.npz: N × T × D sparse tensor containing time-dependent features.

    * N: Number of samples (number of ICU stays) 
    * T: Time step (12 hours of records per sample)
    * D: Dynamic feature dimension (how many features per hour) 

3. test/ARF_12h.csv: Ground truth labels (used for evaluation only).

4. test/X.npz: Test feature set in the same format as training data.

---

## Data usage Notes

To load the features, you need python and the sparse package.

import sparse

X = sparse.load_npz("<url>/X.npz").todense()


To load the labels, use pandas or an alternative csv reader.

import pandas as pd

df = pd.read_csv("<url>/ARF_12h.csv")


---

## Modeling

Each sample is a 12-hour multivariate time series of ICU patient observations, represented as a tensor of shape (12, D).
The goal is to predict whether the patient will develop ARF (1) or not (0) in the following 12 hours.

* **Input**: 12 × D matrix of clinical features
* **Output**: Binary prediction: 0 (no ARF) or 1 (ARF onset)
* **Loss Function**: BCEWithLogitsLoss, CrossEntropyLoss or equivalent
* **Evaluation Metric**: **AUROC** (Area Under the Receiver Operating Characteristic Curve)

Note: Although the output is binary, AUROC evaluates the ranking quality of predicted scores. Therefore, your model should output a confidence score during training, which is then thresholded to produce 0 or 1 for final submission.

---

## Evaluation

### Area Under the Receiver Operating Characteristic curve (AUROC)

The submissions are scored according to the area under the receiver operating characteristic curve. AUROC is defined as:

$$
\text{AUROC} = \frac{1}{|P| \cdot |N|} \sum_{i \in P} \sum_{j \in N} \left[ \mathbb{1}(s_i > s_j) + \frac{1}{2} \cdot \mathbb{1}(s_i = s_j) \right]
$$

AUROC reflects the model's ability to rank positive samples higher than negative ones. A score of 1.0 means perfect discrimination, and 0.5 means random guessing.

### Submission Format

For each `ID'' in the ARF_12h.csv file of the test dataset, you must predict whether ARF will occur (label = 1) or not (label = 0) in the following 12 hours(ARF_LABEL), based on the X.npz (sparse tensor, time-varying feature). The file should contain the following format:

ID,ARF_LABEL
246505,0
291335,0
286713,0
etc.


Note: Although the submission is binary, AUROC evaluates the ranking quality of your model. It is recommended to output probabilities during training and apply a threshold (e.g., 0.5) to convert to binary labels for submission.

---