from rdagent.components.coder.CoSTEER.config import CoSTEERSettings


class DSCoderCoSTEERSettings(CoSTEERSettings):
    """Data Science CoSTEER settings"""

    class Config:
        env_prefix = "DS_Coder_CoSTEER_"

    max_seconds: int = 2400
