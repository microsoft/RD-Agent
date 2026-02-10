"""
AutoRL-Bench Evaluator Base Class

所有 benchmark 评测器的基类，定义统一的评测接口。

开发新 benchmark 时，继承 BaseEvaluator 并实现 run_eval 方法。
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict
from typing_extensions import TypedDict, NotRequired


# ============================================================
# 数据结构定义（Schema）
# ============================================================

class EvalInput(TypedDict):
    """
    评测输入参数
    
    Attributes:
        model_path: 训练后的模型路径（本地目录）
        workspace_path: 工作目录路径
        model_name: 模型名称（用于配置推理参数）
        gpu_count: 可用 GPU 数量
        test_range: 测试数据范围，如 "[:]" 或 "[:100]"
    """
    model_path: str
    workspace_path: str
    model_name: NotRequired[str]
    gpu_count: NotRequired[int]
    test_range: NotRequired[str]


class EvalResult(TypedDict):
    """
    评测输出结果
    
    必须字段:
        benchmark: benchmark 名称
        model_path: 评测的模型路径
        score: 评测分数 (0-100)
        accuracy_summary: 详细指标字典
        
    可选字段:
        eval_type: 评测类型 ("opencompass" / "alfworld" / ...)
        error: 错误信息（评测失败时）
        raw_output: 原始输出日志
    """
    # 必须字段
    benchmark: str
    model_path: str
    score: float
    accuracy_summary: Dict[str, Any]
    
    # 可选字段
    eval_type: NotRequired[str]
    error: NotRequired[str]
    raw_output: NotRequired[str]


# ============================================================
# 抽象基类
# ============================================================

class BaseEvaluator(ABC):
    """
    Benchmark 评测器基类
    
    所有自定义 benchmark 必须继承此类并实现 run_eval 方法。
    
    =====================================================
    最简单的方式：调用 benchmark 自带的评测代码
    =====================================================
    
    大多数 benchmark（如 HumanEval、MBPP、ALFWorld）都有官方评测脚本，
    只需要：
    1. 下载 benchmark repo
    2. 调用它的评测函数
    3. 把结果转成 EvalResult 格式
    
    Example（包装现有评测）:
        class MyBenchmarkEvaluator(BaseEvaluator):
            def __init__(self, config):
                self.config = config
                self.benchmark_id = config.id
            
            def run_eval(self, model_path, workspace_path, **kwargs) -> EvalResult:
                result = self.get_default_result(self.benchmark_id, model_path)
                
                # 1. 调用 benchmark 自带的评测
                from some_benchmark import evaluate  # benchmark 官方库
                raw_result = evaluate(model_path)    # 调用官方评测
                
                # 2. 转换成统一格式
                result["score"] = raw_result["accuracy"] * 100
                result["accuracy_summary"] = raw_result
                return result
    
    =====================================================
    完整示例：自定义评测逻辑
    =====================================================
    
    如果需要完全自定义评测（如交互式环境）：
    
    Example:
        class InteractiveEvaluator(BaseEvaluator):
            def run_eval(self, model_path, workspace_path, **kwargs) -> EvalResult:
                result = self.get_default_result(self.benchmark_id, model_path)
                
                # 1. 加载模型
                model = load_model(model_path)
                
                # 2. 运行评测循环
                success = 0
                for task in tasks:
                    output = model.generate(task.prompt)
                    if task.check(output):
                        success += 1
                
                # 3. 返回结果
                result["score"] = success / len(tasks) * 100
                result["accuracy_summary"] = {"success": success, "total": len(tasks)}
                return result
    """
    
    @abstractmethod
    def run_eval(
        self,
        model_path: str,
        workspace_path: str,
        **kwargs
    ) -> EvalResult:
        """
        执行评测
        
        Args:
            model_path: 训练后的模型路径（本地目录）
            workspace_path: 工作目录路径
            **kwargs: 其他评测参数（见 EvalInput）
            
        Returns:
            EvalResult: 评测结果
        """
        pass
    
    def validate_model(self, model_path: str) -> bool:
        """验证模型路径是否有效"""
        return Path(model_path).exists()
    
    def get_default_result(self, benchmark_name: str, model_path: str) -> EvalResult:
        """返回默认的结果结构"""
        return {
            "benchmark": benchmark_name,
            "model_path": model_path,
            "score": 0.0,
            "accuracy_summary": {},
        }
