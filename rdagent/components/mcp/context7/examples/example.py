"""MCP Unified Interface Usage Example - Error Message and Code Debugging

This example demonstrates how to use the NEW unified MCP interface with PARALLEL service processing:
- query_mcp(query): Auto mode - uses ALL available services in parallel
- query_mcp(query, services="service_name"): Single service mode - direct query
- query_mcp(query, services=["service1", "service2"]): Multi-service mode - parallel processing

Key Features:
- Multiple services' tools are available simultaneously to the LLM
- LLM automatically chooses which service's tools to use
- Automatic handling of unavailable services with warnings
- Configuration-driven based on mcp_config.json
"""

import asyncio
import datetime

from rdagent.components.mcp import query_mcp  # New unified interface
from rdagent.components.mcp import get_service_status
from rdagent.log import rdagent_logger as logger


async def example_single_service():
    """Example 1: Using query_mcp() with a specific service"""

    logger.info(f"🔍 Example 1: Single Service Mode [{datetime.datetime.now()}]")
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

    # Check service availability
    status = get_service_status()
    logger.info(f"🔍 MCP Service Status: {status}")

    # Query documentation using specific MCP service
    logger.info("📋 Querying Context7 documentation service...")
    try:
        result = await query_mcp(
            error_message,
            services="context7",  # Specify a single service
            full_code=full_code,
            max_rounds=5,
            verbose=True,
        )

        if result:
            logger.info("\n✅ Solution obtained:")
            logger.info("-" * 40)
            logger.info(result)
        else:
            logger.error("❌ Failed to get relevant documentation information")

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")


async def example_auto_mode():
    """Example 2: Using query_mcp() in auto mode - ALL services in parallel"""
    logger.info(f"\n🤖 Example 2: Auto Mode - All Services in Parallel [{datetime.datetime.now()}]")
    logger.info("=" * 50)

    # Pandas error example
    pandas_error = """
AttributeError: 'DataFrame' object has no attribute 'append'
The DataFrame.append method was removed in pandas 2.0. 
Use pd.concat() instead.
"""

    logger.info("📋 Querying with ALL available services in parallel...")
    logger.info("🔧 LLM will see tools from all services and choose which to use")
    try:
        # Auto mode: All available services' tools are provided to LLM simultaneously
        result = await query_mcp(pandas_error, verbose=True)  # services=None means use all

        if result:
            logger.info("\n✅ Auto-selected service solution:")
            logger.info("-" * 40)
            logger.info(result)
        else:
            logger.error("❌ Automatic service selection failed to get information")

    except Exception as e:
        logger.error(f"❌ Auto query failed: {e}")


async def example_multi_service():
    """Example 3: Using query_mcp() with multiple services in PARALLEL"""
    logger.info(f"\n🎯 Example 3: Multi-Service Mode - Parallel Processing [{datetime.datetime.now()}]")
    logger.info("=" * 50)

    test_query = "How to implement async/await in Python?"

    logger.info("📋 Querying with multiple specified services in PARALLEL...")
    logger.info("🔧 Tools from all specified services will be available to LLM simultaneously")
    try:
        # Multi-service mode: specified services run in parallel, not sequentially
        result = await query_mcp(
            test_query, services=["context7", "fake_service"], verbose=True  # Both services' tools available at once
        )

        if result:
            logger.info("\n✅ Multi-service query succeeded:")
            logger.info("-" * 40)
            logger.info(result)
        else:
            logger.error("❌ All specified services failed")

    except Exception as e:
        logger.error(f"❌ Multi-service query failed: {e}")


async def main():
    """Main function - demonstrates the complete MCP usage workflow"""
    logger.info("🚀 MCP NEW Unified Interface Usage Examples")
    logger.info("📁 Please ensure mcp_config.json is properly configured")
    logger.info("=" * 60)

    # Run three examples showing different usage modes
    await example_single_service()
    await example_auto_mode()
    await example_multi_service()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
