"""
Handles conversion from a Python file to a Jupyter notebook.
"""

import nbformat

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.share.util import (
    extract_first_section_name_from_code,
    extract_function_body,
    split_code_and_output_into_sections,
)
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import MarkdownAgentOut
from rdagent.utils.agent.tpl import T


class NotebookConverter:
    """
    Builder responsible for writing a Jupyter notebook for a workspace.
    """

    def __init__(self):
        self.notebook_name = "main.ipynb"

    def validate_code_format(self, ws: FBWorkspace) -> str | None:
        """
        Returns None if the code format is valid, otherwise returns an error message.
        """
        code = ws.file_dict["main.py"]
        main_function_body = extract_function_body(code, "main")
        if not main_function_body:
            return "[Error] No main function found in the code. Please ensure that the main function is defined and contains the necessary print statements to divide sections."

        found_section_name = extract_first_section_name_from_code(main_function_body)
        if not found_section_name:
            return "[Error] No sections found in the code. Expected to see 'print(\"Section: <section name>\")' as section dividers. Also make sure that they are actually run and not just comments."

        return None

    def convert(
        self, task: Task, ws: FBWorkspace, stdout: str, use_debug_flag: bool
    ) -> None:
        """
        Build a notebook based on the current progression.
        """
        code = ws.file_dict["main.py"]

        # Handle argparse in the code to ensure it works in a notebook environment
        if "argparse" in code:
            code = (
                "\n".join(
                    [
                        "import sys",
                        "# hack to allow argparse to work in notebook",
                        (
                            'sys.argv = ["main.py", "--debug"]'
                            if use_debug_flag
                            else 'sys.argv = ["main.py"]'
                        ),
                    ]
                )
                + "\n"
                + code
            )

        sections = split_code_and_output_into_sections(code=code, stdout=stdout)
        notebook = nbformat.v4.new_notebook()

        # Use LLM to generate an intro cell for the notebook
        system_prompt = T(".prompts:notebookconverter.system").r()
        user_prompt = T(".prompts:notebookconverter.user").r(
            plan=task.get_task_information(),
            code=code,
        )
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt
        )
        intro_content = MarkdownAgentOut.extract_output(resp)
        notebook.cells.append(nbformat.v4.new_markdown_cell(intro_content))

        for section in sections:
            # Create a markdown cell for the section name and comments
            markdown_content = ""
            if section["name"]:
                markdown_content += f"## {section['name']}\n"
            if section["comments"]:
                markdown_content += f"{section['comments']}\n"
            if markdown_content:
                notebook.cells.append(nbformat.v4.new_markdown_cell(markdown_content))

            # Create a code cell for the section code and output
            if section["code"]:
                cell = nbformat.v4.new_code_cell(section["code"])
                if section["output"]:
                    # For simplicity, treat all output as coming from stdout
                    # TODO: support Jupyter kernel execution and handle outputs appropriately here
                    cell.outputs = [
                        nbformat.v4.new_output(
                            "stream", name="stdout", text=section["output"]
                        )
                    ]
                notebook.cells.append(cell)

        # Save the notebook to the workspace
        with open((ws.workspace_path / self.notebook_name), "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
            logger.info(f"Notebook written to {ws.workspace_path / self.notebook_name}")
