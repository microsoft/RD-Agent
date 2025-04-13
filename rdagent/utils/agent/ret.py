"""
The output of a agent is very important.

We think this part can be shared.
"""

import json
import re
from abc import abstractclassmethod
from typing import Any

from rdagent.utils.agent.tpl import T


class AgentOut:
    json_mode: bool = False  # To get the output, is json_mode required.

    @abstractclassmethod
    def get_spec(cls, **context: Any) -> str:
        raise NotImplementedError(f"Please implement the `get_spec` method")

    @classmethod
    def extract_output(cls, resp: str) -> Any:
        raise resp


class PythonAgentOut(AgentOut):
    @classmethod
    def get_spec(cls):
        return T(".tpl:PythonAgentOut").r()

    @classmethod
    def extract_output(cls, resp: str):
        match = re.search(r".*```[Pp]ython\n(.*)\n```.*", resp, re.DOTALL)
        if match:
            code = match.group(1)
            code = re.sub(r"</?code>", "", code, flags=re.IGNORECASE)
            return code
        return resp


class MarkdownAgentOut(AgentOut):
    @classmethod
    def get_spec(cls):
        return T(".tpl:MarkdownOut").r()

    @classmethod
    def extract_output(cls, resp: str):
        match = re.search(r".*````markdown\n(.*)\n````.*", resp, re.DOTALL)
        if match:
            content = match.group(1)
            return content
        return resp


class BatchEditOut(AgentOut):
    json_mode: bool = True

    @classmethod
    def get_spec(cls, with_del=True):
        return T(".tpl:BatchEditOut").r(with_del=with_del)

    @classmethod
    def extract_output(cls, resp: str):
        return json.loads(resp)


class PythonBatchEditOut(AgentOut):
    @classmethod
    def get_spec(cls, with_del=True):
        return T(".tpl:PythonBatchEditOut").r(with_del=with_del)

    @classmethod
    def extract_output(cls, resp: str):
        code_blocks = {}
        pattern = re.compile(r"```(.*?)\n(.*?)\n```", re.DOTALL)
        matches = pattern.findall(resp)

        for match in matches:
            file_name, code = match
            code_blocks[file_name.strip()] = code.strip()

        return code_blocks
