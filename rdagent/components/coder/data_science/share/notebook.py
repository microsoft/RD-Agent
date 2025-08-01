"""
Handles conversion from a Python file to a Jupyter notebook.
"""

import argparse
from typing import Optional

import nbformat

from rdagent.components.coder.data_science.share.util import (
    extract_first_section_name_from_code,
    extract_function_body,
    split_code_and_output_into_sections,
)
from rdagent.core.experiment import Task
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import MarkdownAgentOut
from rdagent.utils.agent.tpl import T


class NotebookConverter:
    """
    Builder responsible for writing a Jupyter notebook for a workspace.
    """

    def validate_code_format(self, code: str) -> str | None:
        """
        Returns None if the code format is valid, otherwise returns an error message.
        """
        main_function_body = extract_function_body(code, "main")
        if not main_function_body:
            return "[Error] No main function found in the code. Please ensure that the main function is defined and contains the necessary print statements to divide sections."

        found_section_name = extract_first_section_name_from_code(main_function_body)
        if not found_section_name:
            return "[Error] No sections found in the code. Expected to see 'print(\"Section: <section name>\")' as section dividers. Also make sure that they are actually run and not just comments."

        return None

    def convert(
        self,
        task: Optional[Task],
        code: str,
        stdout: str,
        outfile: Optional[str] = None,
        use_debug_flag: bool = False,
    ) -> str:
        """
        Build a notebook based on the current progression.
        """
        # Handle argparse in the code to ensure it works in a notebook environment
        should_handle_argparse = "argparse" in code
        sections = split_code_and_output_into_sections(code=code, stdout=stdout)
        notebook = nbformat.v4.new_notebook()

        # Use LLM to generate an intro cell for the notebook
        if task:
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

        if should_handle_argparse:
            # Remove extra `import sys` since it will be added for argparse handling
            if "import sys\n" in sections[0]["code"]:
                sections[0]["code"] = sections[0]["code"].replace("import sys\n", "")

            # Add sys.argv modification for argparse handling
            sections[0]["code"] = (
                "\n".join(
                    [
                        "import sys",
                        "# hack to allow argparse to work in notebook",
                        ('sys.argv = ["main.py", "--debug"]' if use_debug_flag else 'sys.argv = ["main.py"]'),
                    ]
                )
                + "\n\n"
                + sections[0]["code"].lstrip()
            )

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
                    cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=section["output"])]
                notebook.cells.append(cell)

        # Save the notebook or return it as a string
        if outfile:
            with open((outfile), "w", encoding="utf-8") as f:
                nbformat.write(notebook, f)
                logger.info(f"Notebook written to {outfile}")

        return nbformat.writes(notebook)


if __name__ == "__main__":
    converter = NotebookConverter()
    parser = argparse.ArgumentParser(description="Convert Python code to Jupyter notebook.")
    parser.add_argument("inputfile", type=str, help="Path to the input Python file.")
    parser.add_argument("outfile", type=str, help="Path to the output Notebook file.")
    parser.add_argument(
        "--stdout",
        type=str,
        default="",
        help="Standard output from the code execution.",
    )
    parser.add_argument("--debug", action="store_true", help="Use debug flag to modify sys.argv.")
    args = parser.parse_args()
    converter.convert(
        task=None,
        code=open(args.inputfile, "r").read(),
        stdout=args.stdout,
        outfile=args.outfile,
        use_debug_flag=False,
    )
