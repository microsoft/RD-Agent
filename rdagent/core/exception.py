class CoderException(Exception):
    """
    Exceptions raised when Implementing and running code.
    - start: FactorTask => FactorGenerator
    - end: Get dataframe after execution

    The more detailed evaluation in dataframe values are managed by the evaluator.
    """


class CodeFormatException(CoderException):
    """
    The generated code is not found due format error.
    """


class RuntimeErrorException(CoderException):
    """
    The generated code fail to execute the script.
    """


class NoOutputException(CoderException):
    """
    The code fail to generate output file.
    """


class RunnerException(Exception):
    """
    Exceptions raised when running the code output.
    """


class FactorEmptyException(Exception):
    """
    Exceptions raised when no factor is generated correctly
    """


class ModelEmptyException(Exception):
    """
    Exceptions raised when no model is generated correctly
    """
