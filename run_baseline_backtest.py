#!/usr/bin/env python
"""
Direct baseline backtest runner - bypasses LLM hypothesis generation.
Runs Alpha20 + Top9 factors through the Qlib backtest pipeline.
"""

import json
from pathlib import Path

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.utils.qlib import ALPHA20


TRAIN_START = "2024-01-01"
TRAIN_END = "2024-12-31"
VALID_START = "2025-01-01"
VALID_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2026-03-30"


def run_baseline_backtest():
    FACTOR_PROP_SETTING.train_start = TRAIN_START
    FACTOR_PROP_SETTING.train_end = TRAIN_END
    FACTOR_PROP_SETTING.valid_start = VALID_START
    FACTOR_PROP_SETTING.valid_end = VALID_END
    FACTOR_PROP_SETTING.test_start = TEST_START
    FACTOR_PROP_SETTING.test_end = TEST_END
    
    exp = QlibFactorExperiment(sub_tasks=[])
    
    # 2. Set baseline features (Alpha20 expressions)
    exp.base_features = ALPHA20.copy()
    logger.info(f"Loaded {len(exp.base_features)} Alpha20 features")
    
    # 3. Load Top9 factor codes (comment out for pure Alpha20 baseline)
    USE_TOP9 = True  # Set to False for pure Alpha20 baseline
    if USE_TOP9:
        baseline_dir = Path(__file__).parent / "baseline_features"
        feature_codes = {}
        for py_file in sorted(baseline_dir.glob("*.py")):
            feature_codes[py_file.name] = py_file.read_text()
        exp.base_feature_codes = feature_codes
        logger.info(f"Loaded {len(feature_codes)} factor code files from {baseline_dir}")
    else:
        exp.base_feature_codes = {}
        logger.info("Using pure Alpha20 baseline (no Top9 factors)")
    
    # 4. Create runner and execute
    scen = import_class(FACTOR_PROP_SETTING.scen)()
    runner = QlibFactorRunner(scen)
    
    logger.info("Starting baseline backtest execution...")
    logger.info("Training: 2024-01-01 to 2024-12-31")
    logger.info("Validation: 2025-01-01 to 2025-06-30")
    logger.info("Test: 2025-07-01 to 2026-03-30")
    
    result_exp = runner.develop(exp)
    
    # 5. Output results
    if result_exp.result is not None:
        logger.info("Backtest completed successfully!")
        logger.log_object(result_exp.result, tag="backtest_results")
        
        # Save results
        result_path = Path(__file__).parent / "baseline_backtest_results.json"
        result_dict = result_exp.result.to_dict()
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        logger.info(f"Results saved to {result_path}")
    else:
        logger.error("Backtest failed!")
        logger.error(f"stdout: {result_exp.stdout}")
    
    return result_exp


if __name__ == "__main__":
    run_baseline_backtest()
