# Competition name: playground-series-s4e9

## Overview

**Welcome to the 2024 Kaggle Playground Series!** We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** The goal of this competition is to predict the price of used cars based on various attributes.

## Evaluation

### Root Mean Squared Error (RMSE)

Submissions are scored on the root mean squared error. RMSE is defined as:

$$
\mathrm{RMSE} = \left( \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \right)^{\frac{1}{2}}
$$

where $\hat{y}_i$ is the predicted value and $y_i$ is the original value for each instance $i$.

### Submission File

For each `id` in the test set, you must predict the `price` of the car. The file should contain a header and have the following format:

```
id,price
188533,43878.016
188534,43878.016
188535,43878.016
etc.
```

## Timeline
- **Start Date** - September 1, 2024
- **Entry Deadline** - Same as the Final Submission Deadline
- **Team Merger Deadline** - Same as the Final Submission Deadline
- **Final Submission Deadline** - September 30, 2024

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## About the Tabular Playground Series

The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Prizes
- 1st Place - Choice of Kaggle merchandise
- 2nd Place - Choice of Kaggle merchandise
- 3rd Place - Choice of Kaggle merchandise

**Please note**: In order to encourage more participation from beginners, Kaggle merchandise will only be awarded once per person in this series. If a person has previously won, we'll skip to the next team.

## Citation

Walter Reade and Ashley Chow. Regression of Used Car Prices. https://kaggle.com/competitions/playground-series-s4e9, 2024. Kaggle.

## Dataset Description

The dataset for this competition (both train and test) was generated from a deep learning model trained on the [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset). Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Files

- **train.csv** - the training dataset; `price` is the continuous target
- **test.csv** - the test dataset; your objective is to predict the value of `price` for each row
- **sample_submission.csv** - a sample submission file in the correct format
