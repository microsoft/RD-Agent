"""MCP统一接口使用示例 - 处理错误消息和代码调试

这个示例展示了如何使用新的统一MCP接口来查询Context7文档服务，
帮助解决编程错误和获取相关文档信息。

核心特性：
- 基于mcp_config.json的配置驱动
- 统一的query_mcp()接口
- 支持错误消息和完整代码上下文
- 保留所有优化机制（prompt模板、缓存等）
"""

import asyncio
from pathlib import Path

from rdagent.components.mcp import (
    initialize_mcp_registry,
    is_service_available,
    list_available_mcp_services,
    query_mcp,
    query_mcp_auto,
)


async def example_error_debugging():
    """示例1: 使用MCP解决常见编程错误"""
    print("🔍 示例1: 错误消息调试")
    print("=" * 50)

    # 模拟一个LightGBM GPU错误
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
    # LightGBM参数调优
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
        'device': 'gpu',  # 问题出现在这里 - GPU不可用
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X, label=y)
    gbm = lgb.train(params, train_data, num_boost_round=100)
    
    return gbm.best_score['valid_0']['multi_logloss']
"""

    # 检查Context7服务是否可用
    if not is_service_available("context7"):
        print("⚠️  Context7服务不可用，请检查mcp_config.json配置")
        return

    # 使用统一接口查询文档
    print("📋 查询Context7文档服务...")
    try:
        result = await query_mcp("context7", query=error_message, full_code=full_code, max_rounds=3, verbose=True)

        if result:
            print("\n✅ 获得解决方案:")
            print("-" * 40)
            print(result)
        else:
            print("❌ 未能获取相关文档信息")

    except Exception as e:
        print(f"❌ 查询失败: {e}")


async def example_auto_service_selection():
    """示例2: 自动服务选择"""
    print("\n🤖 示例2: 自动服务选择")
    print("=" * 50)

    # Pandas错误示例
    pandas_error = """
AttributeError: 'DataFrame' object has no attribute 'append'
The DataFrame.append method was removed in pandas 2.0. 
Use pd.concat() instead.
"""

    print("📋 使用自动服务选择查询...")
    try:
        # 让系统自动选择最佳MCP服务
        result = await query_mcp_auto(query=pandas_error, verbose=True)

        if result:
            print("\n✅ 自动选择服务的解决方案:")
            print("-" * 40)
            print(result[:500] + "..." if len(result) > 500 else result)
        else:
            print("❌ 自动服务选择未能获取信息")

    except Exception as e:
        print(f"❌ 自动查询失败: {e}")


async def example_service_management():
    """示例4: 服务管理和状态检查"""
    print("\n⚙️  示例4: MCP服务管理")
    print("=" * 50)

    # 列出可用服务
    services = list_available_mcp_services()
    print(f"📊 可用MCP服务: {services}")

    # 检查特定服务状态
    for service in services:
        available = is_service_available(service)
        status = "✅ 可用" if available else "❌ 不可用"
        print(f"   - {service}: {status}")


async def main():
    """主函数 - 演示完整的MCP使用流程"""
    print("🚀 MCP统一接口使用示例")
    print("📁 请确保mcp_config.json已正确配置")
    print("=" * 60)

    # 1. 初始化MCP注册表（可选，系统会自动初始化）
    try:
        config_path = Path.cwd() / "mcp_config.json"
        if config_path.exists():
            print(f"📋 使用配置文件: {config_path}")
            registry = initialize_mcp_registry(config_path)
            print(f"✅ MCP注册表初始化成功")
        else:
            print("⚠️  未找到mcp_config.json，将使用默认配置")
    except Exception as e:
        print(f"⚠️  配置初始化警告: {e}")

    # 2. 服务状态检查
    await example_service_management()

    # 3. 错误调试示例
    await example_error_debugging()

    # 4. 自动服务选择示例
    await example_auto_service_selection()

    print("\n" + "=" * 60)
    print("🎉 示例运行完成!")
    print("\n💡 双函数接口使用提示:")
    print("1. 确保mcp_config.json中配置了Context7服务")
    print("2. 设置正确的API密钥和模型")
    print("3. 指定服务查询: query_mcp('context7', query='your question')")
    print("4. 自动服务选择: query_mcp_auto(query='your question')")
    print("5. 两个函数职责明确，使用简单直观")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
