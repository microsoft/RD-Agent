"""
The output of a agent is very important.

We think this part can be shared.
"""

import json
import re
from abc import abstractclassmethod
from typing import Any

from rdagent.utils.agent.apply_patch import apply_patch_from_text
from rdagent.utils.agent.tpl import T


class AgentOut:
    json_mode: bool = False  # To get the output, is json_mode required.

    @abstractclassmethod
    def get_spec(cls, **context: Any) -> str:
        raise NotImplementedError("Please implement the `get_spec` method")

    @classmethod
    def extract_output(cls, resp: str) -> Any:
        raise resp


class PythonAgentOut(AgentOut):
    @classmethod
    def get_spec(cls):
        return T(".tpl:PythonAgentOut").r()

    @classmethod
    def extract_output(cls, resp: str):
        # We use lazy mode (.*?) to only extract the first code block in the response.
        match = re.search(r".*```[Pp]ython\n(.*?)\n```.*", resp, re.DOTALL)
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


# TODO: complete the apply_patch() function by @Qizheng
class PythonBatchPatchOut(AgentOut):
    @classmethod
    def get_spec(cls):
        return T(".tpl:PythonBatchPatchOut").r()

    @classmethod
    def apply_patch(cls, patch_text: str) -> str:
        """
        Apply patch text to filesystem using the apply_patch module.

        Args:
            patch_text: The patch text to apply

        Returns:
            Result message from patch application

        Raises:
            DiffError: If patch application fails
        """
        return apply_patch_from_text(patch_text)

    @classmethod
    def extract_output(cls, resp: str) -> str:
        """
        Extract patch content from the response.

        Args:
            resp: The raw response text containing patch

        Returns:
            The extracted patch text
        """
        # Extract patch content from markdown code block if present
        patch_pattern = re.compile(r"```(?:diff|patch)?\n(.*?)\n```", re.DOTALL)
        match = patch_pattern.search(resp)
        if match:
            # Remove trailing whitespace to avoid parsing errors
            return match.group(1).rstrip()

        # If no code block found, return the response as-is
        return resp
