from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.core.conf import ExtendedSettingsConfigDict


class FactorCoSTEERSettings(CoSTEERSettings):
    model_config = ExtendedSettingsConfigDict(env_prefix="FACTOR_CoSTEER_")

    data_folder: str = "git_ignore_folder/factor_implementation_source_data"
    """Path to the folder containing financial data (default is fundamental data in Qlib)"""

    data_folder_debug: str = "git_ignore_folder/factor_implementation_source_data_debug"
    """Path to the folder containing partial financial data (for debugging)"""

    simple_background: bool = False
    """Whether to use simple background information for code feedback"""

    file_based_execution_timeout: int = 120
    """Timeout in seconds for each factor implementation execution"""

    select_method: str = "random"
    """Method for the selection of factors implementation"""

    python_bin: str = "python"
    """Path to the Python binary"""


FACTOR_COSTEER_SETTINGS = FactorCoSTEERSettings()
