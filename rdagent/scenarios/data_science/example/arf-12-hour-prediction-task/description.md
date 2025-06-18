# ARF 12-Hour Prediction Task

## Overview

### Description

Acute Respiratory Failure (ARF) is a life-threatening condition that often develops rapidly in critically ill patients. Accurate early prediction of ARF is crucial in intensive care units (ICUs) to enable timely clinical interventions and resource allocation. In this task, you are asked to build a machine learning model that predicts whether a patient will develop ARF within the next **12 hours**, based on multivariate clinical time series data.

The dataset is extracted from electronic health records (EHRs) and preprocessed using the **FIDDLE** pipeline to generate structured temporal features for each patient.

### Objective

Your goal is to develop a binary classification model that takes a 12-hour time series as input and outputs the probability of ARF onset in the following 12 hours.

---

## Data Description

The dataset is divided into two directories:

* `train/`

  * `ARF_12h.csv`: `ID` & `ARF_ONSET_HOUR` & Binary labels (`ARF_LABEL`) for training samples.
  * `X.npz`: 3D sparse array of shape `(N, T, D)`:

    * `N`: number of training samples
    * `T`: time steps (one per hour)
    * `D`: number of features

* `test/`

  * `ARF_12h.csv`: `ID` & `ARF_ONSET_HOUR`.
  * `X.npz`: Test feature set in the same format as training data.

The `.npz` files store sparse matrices and are loaded using the [`sparse`](https://github.com/pydata/sparse) library. Each matrix is converted to dense format before model input. (DO NOT USE scipy.sparse)
e.g. 
```
import sparse
X = sparse.load_npz("<url>").todense()
```
Then, you can use `X.transpose(0, 2, 1)` to transpose the shape of X from (N, T, D) to (N, D, T)

---

## Modeling

Each sample is a multivariate time series representing 12 hours of clinical observations. Your model should learn temporal and cross-feature dependencies to estimate the likelihood of ARF.

* **Output**: Probability score ∈ \[0, 1]
* **Loss Function**: `CrossEntropyLoss` or equivalent
* **Evaluation Metric**: **AUROC** (Area Under the Receiver Operating Characteristic Curve)

---

## Submission Format
```
ID,ARF_LABEL
0,0.473
1,0.652
2,0.129
...
```
