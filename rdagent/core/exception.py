class CoderError(Exception):
    """
    Exceptions raised when Implementing and running code.
    - start: FactorTask => FactorGenerator
    - end: Get dataframe after execution

    The more detailed evaluation in dataframe values are managed by the evaluator.
    """


class CodeFormatError(CoderError):
    """
    The generated code is not found due format error.
    """


class CustomRuntimeError(CoderError):
    """
    The generated code fail to execute the script.
    """


class NoOutputError(CoderError):
    """
    The code fail to generate output file.
    """


class CustomRunnerError(Exception):
    """
    Exceptions raised when running the code output.
    """


class FactorEmptyError(Exception):
    """
    Exceptions raised when no factor is generated correctly
    """


class ModelEmptyError(Exception):
    """
    Exceptions raised when no model is generated correctly
    """


class KaggleError(Exception):
    """
    Exceptions raised when calling Kaggle API
    """
