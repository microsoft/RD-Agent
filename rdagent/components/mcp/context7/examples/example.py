"""MCP Unified Interface Usage Example - Error Message and Code Debugging

This example demonstrates how to use the unified MCP interface with two main functions:
- query_mcp(): Direct service query with specified MCP service
- query_mcp_auto(): Automatic service selection for optimal results

Key Features:
- Configuration-driven based on mcp_config.json
- Support for error messages and full code context
- Preserves all optimization mechanisms (prompt templates, caching, etc.)
"""

import asyncio
import datetime

from rdagent.components.mcp import (
    query_mcp,
    query_mcp_auto,
)
from rdagent.log import rdagent_logger as logger


async def example_query_mcp():
    """Example 1: Using query_mcp() to solve programming errors with specific service"""

    logger.info(f"🔍 Example 1: query_mcp() - Direct Service Query [{datetime.datetime.now()}]")
    logger.info("=" * 50)

    # Simulate a LightGBM GPU error
    error_message = """### TRACEBACK: Traceback (most recent call last):
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
value_or_values = func(trial)
^^^^^^^^^^^
File "/workspace/main.py", line 226, in <lambda>
study.optimize(lambda trial: lgb_optuna_objective(trial, X_sub, y_sub, num_class, debug),
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/workspace/main.py", line 201, in lgb_optuna_objective
gbm = lgb.train(
^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/lightgbm/engine.py", line 282, in train
booster = Booster(params=params, train_set=train_set)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/lightgbm/basic.py", line 3641, in __init__
_safe_call(
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/lightgbm/basic.py", line 296, in _safe_call
raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))
lightgbm.basic.LightGBMError: No OpenCL device found
### SUPPLEMENTARY_INFO: lgb.train called with device=gpu, gpu_platform_id=0, gpu_device_id=0."""

    full_code = """
import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np

def lgb_optuna_objective(trial, X, y, num_class, debug=False):
    # LightGBM parameter optimization
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'device': 'gpu',  # Problem occurs here - GPU not available
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X, label=y)
    gbm = lgb.train(params, train_data, num_boost_round=100)
    
    return gbm.best_score['valid_0']['multi_logloss']
"""

    # Debug: Check service availability
    from rdagent.components.mcp import get_service_status

    status = get_service_status()
    logger.info(f"🔍 MCP Service Status: {status}")

    # Query documentation using specific MCP service
    logger.info("📋 Querying Context7 documentation service...")
    try:
        result = await query_mcp(
            service_name="context7", query=error_message, full_code=full_code, max_rounds=3, verbose=True
        )

        if result:
            logger.info("\n✅ Solution obtained:")
            logger.info("-" * 40)
            logger.info(result)
        else:
            logger.error("❌ Failed to get relevant documentation information")

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")


async def example_query_mcp_auto():
    """Example 2: Using query_mcp_auto() for automatic service selection"""
    logger.info(f"\n🤖 Example 2: query_mcp_auto() - Automatic Service Selection [{datetime.datetime.now()}]")
    logger.info("=" * 50)

    # Pandas error example
    pandas_error = """
AttributeError: 'DataFrame' object has no attribute 'append'
The DataFrame.append method was removed in pandas 2.0. 
Use pd.concat() instead.
"""

    logger.info("📋 Querying with automatic service selection...")
    try:
        # Let the system automatically choose the best MCP service
        result = await query_mcp_auto(query=pandas_error, verbose=True)

        if result:
            logger.info("\n✅ Auto-selected service solution:")
            logger.info("-" * 40)
            logger.info(result)
        else:
            logger.error("❌ Automatic service selection failed to get information")

    except Exception as e:
        logger.error(f"❌ Auto query failed: {e}")


async def main():
    """Main function - demonstrates the complete MCP usage workflow"""
    logger.info("🚀 MCP Unified Interface Usage Example")
    logger.info("📁 Please ensure mcp_config.json is properly configured")
    logger.info("=" * 60)

    # Run two main examples
    await example_query_mcp()
    await example_query_mcp_auto()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
