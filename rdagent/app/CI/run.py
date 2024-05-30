from __future__ import annotations

"""

"""
import json
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import ndiff
from pathlib import Path
from typing import Dict, List, Tuple, Union, cast, Optional, Any
from tree_sitter import Language, Parser, Tree, Node
import tree_sitter_python

from rdagent.core.evolving_framework import (
    Evaluator,
    EvoAgent,
    EvolvableSubjects,
    EvolvingStrategy,
    EvoStep,
    Feedback,
    Knowledge,
)
from rdagent.oai.llm_utils import APIBackend
from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
import datetime

from .prompts import (
    linting_system_prompt_template,
    session_normal_template,
    session_start_template,
)

py_parser = Parser(Language(tree_sitter_python.language()))

@dataclass
class CIError:
    raw_str: str
    file_path: Path | str
    line: int
    column: int
    code: str
    msg: str
    hint: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__

@dataclass
class CIFeedback(Feedback):
    errors: dict[str, list[CIError]]


@dataclass
class FixRecord:
    skipped_errors: list[CIError]
    directly_fixed_errors: list[CIError]
    manually_fixed_errors: list[CIError]
    manual_instructions: dict[str, list[CIError]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skipped_errors": [error.to_dict() for error in self.skipped_errors],
            "directly_fixed_errors": [error.to_dict() for error in self.directly_fixed_errors],
            "manually_fixed_errors": [error.to_dict() for error in self.manually_fixed_errors],
            "manual_instructions": {
                key: [error.to_dict() for error in errors]
                for key, errors in self.manual_instructions.items()
            },
        }


class CodeFile:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.load()


    @classmethod
    def add_line_number(cls: CodeFile, code: Union[List[str], str], start: int = 1) -> Union[List[str], str]:
        if isinstance(code, str):
            code_lines = code.split("\n")
        else:
            code_lines = code

        lineno_width = len(str(start - 1 + len(code_lines)))
        code_with_lineno = []
        for i, code_line in enumerate(code_lines):
            code_with_lineno.append(f"{i+start: >{lineno_width}} | {code_line}")

        return code_with_lineno if isinstance(code, list) else "\n".join(code_with_lineno)

    @classmethod
    def remove_line_number(cls: CodeFile, code: Union[List[str], str]) -> Union[List[str], str]:
        if isinstance(code, str):
            code_lines = code.split("\n")
        else:
            code_lines = code

        try:
            code_without_lineno = []
            for code_line in code_lines:
                code_without_lineno.append(re.split(r"\| ", code_line, maxsplit=1)[1])
        except IndexError:
            code_without_lineno = ["something went wrong when remove line numbers"] + code_lines

        return code_without_lineno if isinstance(code, list) else "\n".join(code_without_lineno)

    def load(self) -> None:
        code = self.path.read_text(encoding="utf-8")
        self.code_lines = code.split("\n")

        # line numbers
        self.lineno = len(self.code_lines)
        self.lineno_width = len(str(self.lineno))
        self.code_lines_with_lineno = self.add_line_number(self.code_lines)

    def get(self, start: int = 1, end: int | None = None, add_line_number: bool = False, return_list: bool = False) -> list[str] | str:
        """
        Retrieves a portion of the code lines.
        line number starts from 1, return codes in [start, end].

        Args:
            start (int): The starting line number (inclusive). Defaults to 1.
            end (int): The ending line number (inclusive). Defaults to None, which means the last line.
            add_line_number (bool): Whether to include line numbers in the result. Defaults to False.
            return_list (bool): Whether to return the result as a list of lines
                or as a single string. Defaults to False.

        Returns:
            Union[List[str], str]: The code lines as a list of strings or as a
                single string, depending on the value of `return_list`.
        """
        start -= 1
        if start < 0:
            start = 0
        end = self.lineno if end is None else end
        if end <= start: 
            res = []
        res = self.code_lines_with_lineno[start:end] if add_line_number else self.code_lines[start:end]

        return res if return_list else "\n".join(res)

    def apply_changes(self, changes: list[tuple[int, int, str]]) -> None:
        """
        Applies the given changes to the code lines.

        Args:
            changes (List[Tuple[int, int, str]]): A list of tuples representing the changes to be applied.
                Each tuple contains the start line number, end line number, and the new code to be inserted.

        Returns:
            None
        """
        offset = 0
        for start, end, code in changes:
            start -= 1
            if start < 0: start = 0

            new_code = code.split("\n")
            self.code_lines[start+offset:end+offset] = new_code
            offset += len(new_code) - (end - start)

        self.path.write_text("\n".join(self.code_lines), encoding="utf-8")
        self.load()

    def get_code_blocks(self, max_lines: int = 30) -> list[tuple[int, int]]:
        tree = py_parser.parse(bytes("\n".join(self.code_lines), "utf8"))

        def get_blocks_in_node(node: Node, max_lines: int) -> list[tuple[int, int]]:
            if node.type == "assignment":
                return [(node.start_point.row, node.end_point.row + 1)]

            blocks: list[tuple[int, int]] = []
            block: tuple[int, int] | None = None # [start, end), line number starts from 0

            for child in node.children:
                if child.end_point.row + 1 - child.start_point.row > max_lines:
                    if block is not None:
                        blocks.append(block)
                    block = None
                    blocks.extend(get_blocks_in_node(child, max_lines))
                elif block is None:
                    block = (child.start_point.row, child.end_point.row + 1)
                elif child.end_point.row + 1 - block[0] <= max_lines:
                    block = (block[0], child.end_point.row + 1)
                else:
                    blocks.append(block)
                    block = (child.start_point.row, child.end_point.row + 1)
            
            if block is not None:
                blocks.append(block)

            return blocks

        # change line number to start from 1 and [start, end) to [start, end]
        return [(a+1,b) for a,b in get_blocks_in_node(tree.root_node, max_lines)]

    def __str__(self) -> str:
        return f"{self.path}"


class Repo(EvolvableSubjects):
    def __init__(self, project_path: Path | str, **kwargs: Any) -> None:
        self.params = kwargs
        self.project_path = Path(project_path)
        git_ignored_output = subprocess.check_output(
            "git status --ignored -s",
            shell=True,
            cwd=project_path,
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
        git_ignored_files = [
            (self.project_path / Path(line[3:])).resolve()
            for line in git_ignored_output.split("\n")
            if line.startswith("!!")
        ]

        files = [
            file
            for file in self.project_path.glob("**/*")
            if file.is_file()
            and not any(str(file).startswith(str(path)) for path in git_ignored_files)
            and ".git/" not in str(file)
            and file.suffix == ".py"
        ]
        self.files = {file: CodeFile(file) for file in files}

        self.fix_records: dict[str, FixRecord] | None = None


@dataclass
class RuffRule:
    """
    {
        "name": "missing-trailing-comma",
        "code": "COM812",
        "linter": "flake8-commas",
        "summary": "Trailing comma missing",
        "message_formats": [
            "Trailing comma missing"
        ],
        "fix": "Fix is always available.",
        "explanation": "## What it does\nChecks for the absence of trailing commas.\n\n## Why is this bad?\nThe presence of a trailing comma can reduce diff size when parameters or\nelements are added or removed from function calls, function definitions,\nliterals, etc.\n\n## Example\n```python\nfoo = {\n    \"bar\": 1,\n    \"baz\": 2\n}\n```\n\nUse instead:\n```python\nfoo = {\n    \"bar\": 1,\n    \"baz\": 2,\n}\n```\n",
        "preview": false
    }
    """
    name: str
    code: str
    linter: str
    summary: str
    message_formats: list[str]
    fix: str
    explanation: str
    preview: bool


class RuffEvaluator(Evaluator):
    """The error message are generated by
    `python -m ruff .  --exclude FinCo,finco,fincov1 --ignore ANN101,TCH003,D,ERA001`
    """

    def __init__(self, command: Optional[str] = None) -> None:
        if command is None:
            self.command = "ruff check . --no-fix --output-format full"
        else:
            self.command = command


    def explain_rule(self, error_code: str) -> RuffRule:
        explain_command = "ruff rule {error_code} --output-format json"
        try:
            out = subprocess.check_output(
                explain_command.format(error_code=error_code),
                shell=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            out = e.output

        return RuffRule(**json.loads(out.decode()))


    def evaluate(self, evo: Repo, **kwargs) -> CIFeedback:
        """Simply run ruff to get the feedbacks."""
        try:
            out = subprocess.check_output(
                self.command,
                shell=True,
                cwd=evo.project_path,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            out = e.output

        """ruff output format:
        src/finco/cli.py:9:5: ANN201 Missing return type annotation for public function `main`
        |
        9 | def main(prompt=None):
        |     ^^^^ ANN201
        10 |     load_dotenv(verbose=True, override=True)
        11 |     wm = WorkflowManager()
        |
        = help: Add return type annotation: `None`
        """

        # extract error info
        pattern = r"(([^\n]*):(\d+):(\d+): (\w+) ([^\n]*)\n(.*?))\n\n"
        matches = re.findall(pattern, out.decode(), re.DOTALL)

        errors = defaultdict(list)
        for match in matches:
            raw_str, file_path, line_number, column_number, error_code, error_message, error_hint = match
            error = CIError(raw_str=raw_str,
                            file_path=file_path,
                            line=int(line_number),
                            column=int(column_number),
                            code=error_code,
                            msg=error_message,
                            hint=error_hint)
            errors[file_path].append(error)

        return CIFeedback(errors=errors)

class MypyEvaluator(Evaluator):

    def __init__(self, command: str | None = None) -> None:
        if command is None:
            self.command = "mypy . --explicit-package-bases"
        else:
            self.command = command

    def evaluate(self, evo: Repo, **kwargs) -> CIFeedback:
        try:
            out = subprocess.check_output(
                self.command,
                shell=True,
                cwd=evo.project_path,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            out = e.output

        return CIFeedback(cast(str, out).decode("utf-8"))


class CIEvoStr(EvolvingStrategy):

    def evolve(
        self,
        evo: Repo,
        evolving_trace: Optional[List[EvoStep]] = None,
        knowledge_l: Optional[List[Knowledge]] = None,
        **kwargs: Any,
    ) -> Repo:
        api = APIBackend()
        system_prompt = linting_system_prompt_template.format(language="Python")

        if len(evolving_trace) > 0:
            last_feedback: CIFeedback = evolving_trace[-1].feedback
            fix_records: dict[str, FixRecord] = defaultdict(lambda: FixRecord([], [], [], defaultdict(list)))
            # iterate by file
            for file_path, errors in last_feedback.errors.items():
                if "CI/run.py" not in file_path:
                    break
                print(Rule(f"[cyan]Fixing {file_path}[/cyan]", style="bold cyan", align="left", characters="."))

                file = evo.files[evo.project_path / Path(file_path)]

                changes: list[tuple[int, int, str]] = []

                # check if the file needs to add `from __future__ import annotations`
                # need to add rules here for different languages/tools
                # TODO @bowen: current way of handling errors like 'Add import statement' may be not good
                for error in errors:
                    if error.code in ("FA100", "FA102"):
                        changes.append((0, 0, "from __future__ import annotations\n"))
                        break
                errors = [e for e in errors if e.code not in ("FA100", "FA102")]

                # Group errors by code blocks
                groups: list[tuple[int, int, list[CIError]]] = []
                error_p = 0
                for start_line, end_line in file.get_code_blocks(max_lines=30):
                    group_errors = []
                    while error_p < len(errors) and start_line <= errors[error_p].line <= end_line:
                        group_errors.append(errors[error_p])
                        error_p += 1
                    if group_errors:
                        groups.append((start_line, end_line, group_errors))

                # generate changes
                for group_id, (start_line, end_line, group_errors) in enumerate(groups, start=1):
                    session = api.build_chat_session(session_system_prompt=system_prompt)
                    session.build_chat_completion(session_start_template.format(code=file.get(add_line_number=True)))

                    print(f"[yellow]Fixing part {group_id}...[/yellow]\n")

                    front_context = file.get(start_line-3, start_line-1)
                    rear_context = file.get(end_line+1, end_line+3)
                    front_context_with_lineno = file.get(start_line-3, start_line-1, add_line_number=True)
                    rear_context_with_lineno = file.get(end_line+1, end_line+3, add_line_number=True)


                    code_snippet_with_lineno = file.get(start_line, end_line, add_line_number=True, return_list=False)
                    code_snippet_lines = file.get(start_line, end_line, add_line_number=False, return_list=True)

                    errors_str = "\n".join([f"{error.raw_str}\n" for error in group_errors])

                    # print errors
                    printed_errors_str = "\n".join(
                        [f"{error.line: >{file.lineno_width}}: {error.code: >8} {error.msg}" for error in group_errors],
                    )
                    print(
                        Panel.fit(
                            Syntax(printed_errors_str, lexer="python", background_color="default"),
                            title=f"{len(group_errors)} Errors",
                        ),
                    )

                    # print original code
                    table = Table(show_header=False, box=None)
                    table.add_column()
                    table.add_row(Syntax(front_context_with_lineno, lexer="python", background_color="default"))
                    table.add_row(Rule(style="dark_orange"))
                    table.add_row(Syntax(code_snippet_with_lineno, lexer="python", background_color="default"))
                    table.add_row(Rule(style="dark_orange"))
                    table.add_row(Syntax(rear_context_with_lineno, lexer="python", background_color="default"))
                    print(Panel.fit(table, title="Original Code"))

                    # ask LLM to repair current code snippet
                    user_prompt = session_normal_template.format(
                        code=code_snippet_with_lineno,
                        lint_info=errors_str,
                        start_line=start_line,
                        end_line=end_line,
                        start_lineno=start_line,
                    )
                    res = session.build_chat_completion(user_prompt)

                    manual_fix_flag = False

                    while True:
                        try:
                            new_code = re.search(r".*```[Pp]ython\n(.*)\n```.*", res, re.DOTALL).group(1)
                        except Exception:
                            print(f"[red]Error when extract codes[/red]:\n {res}")

                        new_code = CodeFile.remove_line_number(new_code)
                        # print repair status (code diff)
                        diff = ndiff(code_snippet_lines, new_code.split("\n"))
                        table = Table(show_header=False, box=None)
                        table.add_column()

                        # add 2 spaces to align with diff format
                        front_context = re.sub(r"^", "  ", front_context, flags=re.MULTILINE)
                        rear_context = re.sub(r"^", "  ", rear_context, flags=re.MULTILINE)

                        table.add_row(Syntax(front_context, lexer="python", background_color="default"))
                        table.add_row(Rule(style="dark_orange"))
                        for i in diff:
                            if i.startswith("+"):
                                table.add_row(Text(i, style="green"))
                            elif i.startswith("-"):
                                table.add_row(Text(i, style="red"))
                            elif i.startswith("?"):
                                table.add_row(Text(i, style="yellow"))
                            else:
                                table.add_row(Syntax(i, lexer="python", background_color="default"))
                        table.add_row(Rule(style="dark_orange"))
                        table.add_row(Syntax(rear_context, lexer="python", background_color="default"))
                        print(Panel.fit(table, title="Repair Status"))

                        operation = input("Input your operation: ")
                        if operation in ("s", "skip"):
                            fix_records[file_path].skipped_errors.extend(group_errors)
                            break
                        if operation in ("a", "apply"):
                            if manual_fix_flag:
                                fix_records[file_path].manually_fixed_errors.extend(group_errors)
                            else:
                                fix_records[file_path].directly_fixed_errors.extend(group_errors)

                            changes.append((start_line, end_line, new_code))
                            break
                        
                        manual_fix_flag = True
                        fix_records[file_path].manual_instructions[operation].extend(group_errors)
                        res = session.build_chat_completion("There are some problems with the code you provided, "
                                            "please follow the instruction below to fix it again and return.\n"
                                            f"Instruction: {operation}")

                # apply changes
                file.apply_changes(changes)

            evo.fix_records = fix_records

        return evo


DIR = None
while True:
    DIR = Prompt.ask("Please input the project directory")
    DIR = Path(DIR)
    if DIR.exists():
        break
    else:
        print("Invalid directory. Please try again.")

start_time = time.time()
start_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%m%d%H%M")

evo = Repo(DIR)
eval = RuffEvaluator()
estr = CIEvoStr()
rag = None  # RAG is not enable firstly.
ea = EvoAgent(estr, rag=rag)
ea.step_evolving(evo, eval)
while True:
    print(Rule(f"Round {len(ea.evolving_trace)} repair", style="blue"))
    evo: Repo = ea.step_evolving(evo, eval)

    fix_records = evo.fix_records
    filename = f"{DIR.name}_{start_timestamp}_fix_records_{len(ea.evolving_trace)}.json"
    with Path(filename).open("w") as file:
        json.dump([v.to_dict() for k,v in fix_records.items()], file, indent=4)

    # Count the number of skipped errors
    skipped_errors_count = 0
    directly_fixed_errors_count = 0
    manually_fixed_errors_count = 0
    skipped_errors_code_count = defaultdict(int)
    directly_fixed_errors_code_count = defaultdict(int)
    manually_fixed_errors_code_count = defaultdict(int)
    for record in fix_records.values():
        skipped_errors_count += len(record.skipped_errors)
        directly_fixed_errors_count += len(record.directly_fixed_errors)
        manually_fixed_errors_count += len(record.manually_fixed_errors)
        for error in record.skipped_errors:
            skipped_errors_code_count[error.code] += 1
        for error in record.directly_fixed_errors:
            directly_fixed_errors_code_count[error.code] += 1
        for error in record.manually_fixed_errors:
            manually_fixed_errors_code_count[error.code] += 1

    skipped_errors_statistics = ""
    directly_fixed_errors_statistics = ""
    manually_fixed_errors_statistics = ""
    for code, count in sorted(skipped_errors_code_count.items(), key=lambda x: x[1], reverse=True):
        skipped_errors_statistics += f"{count: >5} {code: >10} {eval.explain_rule(code).summary}\n"
    for code, count in sorted(directly_fixed_errors_code_count.items(), key=lambda x: x[1], reverse=True):
        directly_fixed_errors_statistics += f"{count: >5} {code: >10} {eval.explain_rule(code).summary}\n"
    for code, count in sorted(manually_fixed_errors_code_count.items(), key=lambda x: x[1], reverse=True):
        manually_fixed_errors_statistics += f"{count: >5} {code: >10} {eval.explain_rule(code).summary}\n"

    # Create a table to display the counts and ratios
    table = Table(title="Error Fix Statistics")
    table.add_column("Type")
    table.add_column("Statistics")
    table.add_column("Count")
    table.add_column("Ratio")

    total_errors_count = skipped_errors_count + directly_fixed_errors_count + manually_fixed_errors_count
    table.add_row("Total Errors", "", str(total_errors_count), "")
    table.add_row("Skipped Errors", skipped_errors_statistics, 
                   str(skipped_errors_count), 
                   f"{skipped_errors_count / total_errors_count:.2%}")
    table.add_row("Directly Fixed Errors", directly_fixed_errors_statistics, 
                   str(directly_fixed_errors_count), 
                   f"{directly_fixed_errors_count / total_errors_count:.2%}")
    table.add_row("Manually Fixed Errors", manually_fixed_errors_statistics, 
                   str(manually_fixed_errors_count), 
                   f"{manually_fixed_errors_count / total_errors_count:.2%}")

    print(table)
    operation = Prompt.ask("Start next round? (y/n)", choices=["y", "n"])
    if operation == "n":
        break


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

""" Please commit it by hand... and then run the next round
git add -u
git commit --no-verify  -v
"""
