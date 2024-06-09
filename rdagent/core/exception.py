class ImplementRunException(Exception):
    """
    Exceptions raised when Implementing and running code.
    - start: FactorImplementationTask => FactorGenerator
    - end: Get dataframe after execution

    The more detailed evaluation in dataframe values are managed by the evaluator.
    """


class CodeFormatException(ImplementRunException):
    """
    The generated code is not found due format error.
    """


class RuntimeErrorException(ImplementRunException):
    """
    The generated code fail to execute the script.
    """


class NoOutputException(ImplementRunException):
    """
    The code fail to generate output file.
    """
