"""
FT 场景日志数据加载模块
从文件系统读取 pkl 日志文件，解析为结构化数据
"""

import re
from pathlib import Path
from typing import Any

from rdagent.log.storage import FileStorage


def extract_loop_id(tag: str) -> int | None:
    """从 tag 中提取 Loop ID"""
    match = re.search(r"Loop_(\d+)", tag)
    return int(match.group(1)) if match else None


def extract_evo_id(tag: str) -> int | None:
    """从 tag 中提取 evo_loop ID"""
    match = re.search(r"evo_loop_(\d+)", tag)
    return int(match.group(1)) if match else None


def get_valid_sessions(log_folder: Path) -> list[str]:
    """获取所有有效的会话目录（按时间倒序）"""
    if not log_folder.exists():
        return []

    sessions = []
    for d in log_folder.iterdir():
        if d.is_dir() and d.joinpath("__session__").exists():
            sessions.append(d.name)

    return sorted(sessions, reverse=True)


def load_ft_data(log_path: Path) -> dict[str, Any]:
    """
    加载 FT 日志数据

    返回结构：
    {
        "scenario": LLMFinetuneScen | None,
        "settings": dict,
        "loops": {
            0: {
                "experiment": FTExperiment | None,
                "evo_loops": {
                    0: {"code": list, "feedback": Any},
                    ...
                },
                "feedback": HypothesisFeedback | None,
                "time_info": dict
            },
            ...
        }
    }
    """
    data: dict[str, Any] = {
        "scenario": None,
        "settings": {},
        "loops": {},
    }

    storage = FileStorage(log_path)

    for msg in storage.iter_msg():
        tag = msg.tag
        content = msg.content

        if not tag:
            continue

        # 解析 scenario
        if tag == "scenario":
            data["scenario"] = content
            continue

        # 解析 settings
        if "SETTINGS" in tag:
            data["settings"][tag] = content
            continue

        # 解析 Loop 数据
        loop_id = extract_loop_id(tag)
        if loop_id is None:
            continue

        if loop_id not in data["loops"]:
            data["loops"][loop_id] = {
                "experiment": None,
                "evo_loops": {},
                "feedback": None,
                "runner_result": None,
                "time_info": {},
            }

        loop_data = data["loops"][loop_id]

        # experiment generation
        if "experiment generation" in tag:
            loop_data["experiment"] = content
            continue

        # evolving code
        if "evolving code" in tag:
            evo_id = extract_evo_id(tag)
            if evo_id is not None:
                if evo_id not in loop_data["evo_loops"]:
                    loop_data["evo_loops"][evo_id] = {"code": None, "feedback": None}
                loop_data["evo_loops"][evo_id]["code"] = content
            continue

        # evolving feedback - 注意 tag 中有空格
        if "evolving feedback" in tag:
            evo_id = extract_evo_id(tag)
            if evo_id is not None:
                if evo_id not in loop_data["evo_loops"]:
                    loop_data["evo_loops"][evo_id] = {"code": None, "feedback": None}
                # 直接存储，不依赖 bool 值
                evo_entry = loop_data["evo_loops"][evo_id]
                evo_entry["feedback"] = content
            continue

        # feedback (final)
        if "feedback.feedback" in tag or (tag.endswith(".feedback") and "evo_loop" not in tag):
            loop_data["feedback"] = content
            continue

        # runner result (Full Train)
        if "runner result" in tag:
            loop_data["runner_result"] = content
            continue

        # time_info
        if "time_info" in tag:
            stage = "direct_exp_gen" if "direct_exp_gen" in tag else "coding" if "coding" in tag else "feedback"
            loop_data["time_info"][stage] = content

    return data
