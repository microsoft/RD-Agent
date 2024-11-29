"""
Beyond previous tests
- 
"""


# Below are unit tests for testing the specification of the implemented model ------------------
#
class XXX1SpecEval:
    """
    Motivation case:
    - Simplest case, we already split the data into train_data, valid_data, and test_data. We require the model to learn (optionally validate on valid data), and infer on test data.

    Test workflow:
    - Build train, valid, and test data to run it, and test the output (e.g., shape, value, etc.)
    """


class XXX2SpecEval:
    """
    Based on XXX1SpecEval, but considering the following case:
    
    Motivation case:
    - Sometimes we don't need validation (e.g., simple models not prone to overfitting, or data is too scarce to split).

    Test workflow:
    - Build train and test data to run it, and test the output (e.g., shape, value, etc.)
    - valid_data == None
    """


class XXX3SpecEval:
    """
    Motivation case:
    - We need to tune hyperparameters.

    Test workflow:
    - Input:
        - Build train and valid data
        - test == None
        - Hyperparameters are not blank
    - Output:
        - The early stop hyperparameters must be returned
    """


class XXX4SpecEval:
    """
    Motivation case:
    - After obtaining good hyperparameters, we retrain the model.
    
    Test workflow:
    - Test1: Since we have already tested it in XXX2SpecEval, we'll focus on another aspect.
        - Input:
            - Build train and test data
            - valid == None
            - Previous good hyperparameters (a parameter representing early stop)
    - Test2: Ensure the hyperparameters are 1) being used, and 2) the model remains stable.
        - Different hyperparameters will yield different results
        - Same hyperparameters will yield the same results
    """
