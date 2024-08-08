"""CLI entrance for all rdagent application."""
import fire
from rdagent.app.qlib_rd_loop.factor import main as fin_factor
from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
from rdagent.app.qlib_rd_loop.model import main as fin_model
from rdagent.app.data_mining.model import main as med_model
from rdagent.app.general_model.general_model import extract_models_and_implement as general_model


if __name__ == "__main__":
    fire.Fire({
        "fin_factor": fin_factor,
        "fin_factor_report": fin_factor_report,
        "fin_model": fin_model,
        "med_model": med_model,
        "gemeral_model": general_model,
    })
