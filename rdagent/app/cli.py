"""
CLI entrance for all rdagent application.

This will 
- make rdagent a nice entry and
- autoamtically load dotenv
"""
import fire
from dotenv import load_dotenv

from rdagent.app.data_mining.model import main as med_model
from rdagent.app.general_model.general_model import (
    extract_models_and_implement as general_model,
)
from rdagent.app.qlib_rd_loop.factor import main as fin_factor
from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
from rdagent.app.qlib_rd_loop.model import main as fin_model

load_dotenv()


def app():
    fire.Fire(
        {
            "fin_factor": fin_factor,
            "fin_factor_report": fin_factor_report,
            "fin_model": fin_model,
            "med_model": med_model,
            "general_model": general_model,
        }
    )
