from __future__ import annotations

import datetime
import json
import re
import shlex
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import ndiff
from pathlib import Path
from typing import Any, Literal

import tree_sitter_python
from rich import print
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from tree_sitter import Language, Node, Parser

from rdagent.core.evaluation import Evaluator
from rdagent.core.evolving_agent import EvoAgent
from rdagent.core.evolving_framework import (
    EvolvableSubjects,
    EvolvingStrategy,
    EvoStep,
    Feedback,
    Knowledge,
)
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import APIBackend

py_parser = Parser(Language(tree_sitter_python.language()))
CI_prompts = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


@dataclass
class CIError:
    raw_str: str
    file_path: Path | str
    line: int
    column: int
    code: str
    msg: str
    hint: str
    checker: Literal["ruff", "mypy"]

    def to_dict(self) -> dict[str, object]:
        return self.__dict__

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line}:{self.column}: {self.code} {self.msg}\n{self.hint}".strip()


@dataclass
class CIFeedback(Feedback):
    errors: dict[str, list[CIError]]

    def statistics(self) -> dict[Literal["ruff", "mypy"], dict[str, int]]:
        error_counts = defaultdict(lambda: defaultdict(int))
        for file_errors in self.errors.values():
            for error in file_errors:
                error_counts[error.checker][error.code] += 1
        return error_counts


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
                key: [error.to_dict() for error in errors] for key, errors in self.manual_instructions.items()
            },
        }


class CodeFile:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.load()

    @classmethod
    def add_line_number(cls: CodeFile, code: list[str] | str, start: int = 1) -> list[str] | str:
        code_lines = code.split("\n") if isinstance(code, str) else code

        lineno_width = len(str(start - 1 + len(code_lines)))
        code_with_lineno = []
        for i, code_line in enumerate(code_lines):
            code_with_lineno.append(f"{i+start: >{lineno_width}} | {code_line}")

        return code_with_lineno if isinstance(code, list) else "\n".join(code_with_lineno)

    @classmethod
    def remove_line_number(cls: CodeFile, code: list[str] | str) -> list[str] | str:
        code_lines = code.split("\n") if isinstance(code, str) else code

        try:
            code_without_lineno = [re.split(r"\| ", code_line, maxsplit=1)[1] for code_line in code_lines]
        except IndexError:
            code_without_lineno = ["something went wrong when remove line numbers", *code_lines]

        return code_without_lineno if isinstance(code, list) else "\n".join(code_without_lineno)

    def load(self) -> None:
        code = self.path.read_text(encoding="utf-8")
        self.code_lines = code.split("\n")

        # line numbers
        self.lineno = len(self.code_lines)
        self.lineno_width = len(str(self.lineno))
        self.code_lines_with_lineno = self.add_line_number(self.code_lines)

    def get(
        self,
        start: int = 1,
        end: int | None = None,
        *,
        add_line_number: bool = False,
        return_list: bool = False,
    ) -> list[str] | str:
        """
        Retrieves a portion of the code lines.
        line number starts from 1, return codes in [start, end].

        Args:
            start (int): The starting line number (inclusive). Defaults to 1.
            end (int | None): The ending line number (inclusive). Defaults to None, which means the last line.
            add_line_number (bool): Whether to include line numbers in the result. Defaults to False.
            return_list (bool): Whether to return the result as a list of lines
                or as a single string. Defaults to False.

        Returns:
            list[str] | str: The code lines as a list of strings or as a
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
            # starts from 1  -->  starts from 0
            adjusted_start = max(start - 1, 0)

            new_code = code.split("\n")
            self.code_lines[adjusted_start + offset : end + offset] = new_code
            offset += len(new_code) - (end - adjusted_start)

        self.path.write_text("\n".join(self.code_lines), encoding="utf-8")
        self.load()

    def get_code_blocks(self, max_lines: int = 30) -> list[tuple[int, int]]:
        tree = py_parser.parse(bytes("\n".join(self.code_lines), "utf8"))

        def get_blocks_in_node(node: Node, max_lines: int) -> list[tuple[int, int]]:
            if node.type == "assignment":
                return [(node.start_point.row, node.end_point.row + 1)]

            blocks: list[tuple[int, int]] = []
            block: tuple[int, int] | None = None  # [start, end), line number starts from 0

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
        return [(a + 1, b) for a, b in get_blocks_in_node(tree.root_node, max_lines)]

    def __str__(self) -> str:
        return f"{self.path}"


class Repo(EvolvableSubjects):
    def __init__(self, project_path: Path | str, excludes: list[Path] | None = None, **kwargs: Any) -> None:
        if excludes is None:
            excludes = []
        self.params = kwargs
        self.project_path = Path(project_path)

        excludes = [self.project_path / path for path in excludes]

        git_ignored_output = subprocess.check_output(
            ["/usr/bin/git", "status", "--ignored", "-s"],  # noqa: S603
            cwd=str(self.project_path),
            stderr=subprocess.STDOUT,
            text=True,
        )
        git_ignored_files = [
            (self.project_path / Path(line[3:])).resolve()
            for line in git_ignored_output.split("\n")
            if line.startswith("!!")
        ]

        excludes.extend(git_ignored_files)

        files = [
            file
            for file in self.project_path.glob("**/*")
            if file.is_file()
            and not any(str(file).startswith(str(path)) for path in excludes)
            and ".git/" not in str(file)
            and file.suffix == ".py"
        ]
        self.files = {file: CodeFile(file) for file in files}

        self.fix_records: dict[str, FixRecord] | None = None


@dataclass
class RuffRule:
    """
    Example:
    {
        "name": "missing-trailing-comma",
        "code": "COM812",
        "linter": "flake8-commas",
        "summary": "Trailing comma missing",
        "message_formats": [
            "Trailing comma missing"
        ],
        "fix": "Fix is always available.",
        "explanation": "...",
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
    """
    The error message are generated by command
    """

    def __init__(self, command: str | None = None) -> None:
        if command is None:
            self.command = "ruff check . --output-format full"
        else:
            self.command = command

    @staticmethod
    def explain_rule(error_code: str) -> RuffRule:
        explain_command = f"ruff rule {error_code} --output-format json"
        try:
            out = subprocess.check_output(
                shlex.split(explain_command),  # noqa: S603
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            out = e.output

        return RuffRule(**json.loads(out))

    def evaluate(self, evo: Repo, **kwargs: dict) -> CIFeedback:
        """Simply run ruff to get the feedbacks."""
        try:
            out = subprocess.check_output(
                shlex.split(self.command),  # noqa: S603
                cwd=evo.project_path,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            out = e.output

        """ruff output format:
        rdagent/cli.py:9:5: ANN201 Missing return type annotation for public function `main`
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
        matches = re.findall(pattern, out, re.DOTALL)

        errors = defaultdict(list)

        for match in matches:
            raw_str, file_path, line_number, column_number, error_code, error_message, error_hint = match

            # TODO @bowen: filter these files when running the check command
            if evo.project_path / Path(file_path) not in evo.files:
                continue
            error = CIError(
                raw_str=raw_str,
                file_path=file_path,
                line=int(line_number),
                column=int(column_number),
                code=error_code,
                msg=error_message,
                hint=error_hint,
                checker="ruff",
            )

            errors[file_path].append(error)

        return CIFeedback(errors=errors)


class MypyEvaluator(Evaluator):
    def __init__(self, command: str | None = None) -> None:
        if command is None:
            self.command = "mypy . --pretty --no-error-summary --show-column-numbers"
        else:
            self.command = command

    def evaluate(self, evo: Repo, **kwargs: dict) -> CIFeedback:
        try:
            out = subprocess.check_output(
                shlex.split(self.command),  # noqa: S603
                cwd=evo.project_path,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            out = e.output

        errors = defaultdict(list)

        out = re.sub(r"([^\n]*?:\d+:\d+): error:", r"\n\1: error:", out)
        out += "\n"
        pattern = r"(([^\n]*?):(\d+):(\d+): error:(.*?)\s\[([\w-]*?)\]\s(.*?))\n\n"
        for match in re.findall(pattern, out, re.DOTALL):
            raw_str, file_path, line_number, column_number, error_message, error_code, error_hint = match
            error_message = error_message.strip().replace("\n", " ")
            if re.match(r".*[^\n]*?:\d+:\d+: note:.*", error_hint, flags=re.DOTALL) is not None:
                error_hint_position = re.split(
                    pattern=r"[^\n]*?:\d+:\d+: note:",
                    string=error_hint,
                    maxsplit=1,
                    flags=re.DOTALL,
                )[0]
                error_hint_help = re.findall(r"^.*?:\d+:\d+: note: (.*)$", error_hint, flags=re.MULTILINE)
                error_hint_help = "\n".join(error_hint_help)
                error_hint = f"{error_hint_position}\nHelp:\n{error_hint_help}"

            if evo.project_path / Path(file_path) not in evo.files:
                continue
            error = CIError(
                raw_str=raw_str,
                file_path=file_path,
                line=int(line_number),
                column=int(column_number),
                code=error_code,
                msg=error_message,
                hint=error_hint,
                checker="mypy",
            )

            errors[file_path].append(error)

        return CIFeedback(errors=errors)


class MultiEvaluator(Evaluator):
    def __init__(self, *evaluators: Evaluator) -> None:
        self.evaluators = evaluators

    def evaluate(self, evo: Repo, **kwargs: dict) -> CIFeedback:
        all_errors = defaultdict(list)
        for evaluator in self.evaluators:
            feedback: CIFeedback = evaluator.evaluate(evo, **kwargs)
            for file_path, errors in feedback.errors.items():
                all_errors[file_path].extend(errors)

        # sort errors by position
        for file_path in all_errors:
            all_errors[file_path].sort(key=lambda x: (x.line, x.column))

        return CIFeedback(errors=all_errors)


class CIEvoStr(EvolvingStrategy):
    def evolve(  # noqa: C901, PLR0912, PLR0915
        self,
        evo: Repo,
        evolving_trace: list[EvoStep] | None = None,
        knowledge_l: list[Knowledge] | None = None,
        **kwargs: dict,
    ) -> Repo:
        @dataclass
        class CodeFixGroup:
            start_line: int
            end_line: int
            errors: list[CIError]
            session_id: str
            responses: list[str]

        api = APIBackend()
        system_prompt = CI_prompts["linting_system_prompt_template"].format(language="Python")

        if len(evolving_trace) > 0:
            last_feedback: CIFeedback = evolving_trace[-1].feedback

            # print statistics
            checker_error_counts = {
                checker: sum(c_statistics.values()) for checker, c_statistics in last_feedback.statistics().items()
            }
            print(
                f"Found [red]{sum(checker_error_counts.values())}[/red] errors, "
                "including: "
                + ", ".join(
                    f"[red]{count}[/red] [magenta]{checker}[/magenta] errors"
                    for checker, count in checker_error_counts.items()
                ),
            )

            fix_records: dict[str, FixRecord] = defaultdict(
                lambda: FixRecord([], [], [], defaultdict(list)),
            )

            # Group errors by code blocks
            fix_groups: dict[str, list[CodeFixGroup]] = defaultdict(list)
            changes: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
            for file_path, errors in last_feedback.errors.items():
                file = evo.files[evo.project_path / Path(file_path)]

                # check if the file needs to add `from __future__ import annotations`
                # need to add rules here for different languages/tools
                # TODO @bowen: current way of handling errors like 'Add import statement' may be not good
                for error in errors:
                    if error.code in ("FA100", "FA102"):
                        changes[file_path].append((1, 1, "from __future__ import annotations\n"))
                        break

                # Group errors by code blocks
                error_p = 0
                for start_line, end_line in file.get_code_blocks(max_lines=30):
                    group_errors: list[CIError] = []

                    # collect errors in the same code block
                    while error_p < len(errors) and start_line <= errors[error_p].line <= end_line:
                        if errors[error_p].code not in ("FA100", "FA102"):
                            group_errors.append(errors[error_p])
                        error_p += 1

                    # process errors in the code block
                    if group_errors:
                        session = api.build_chat_session(session_system_prompt=system_prompt)
                        session_id = session.get_conversation_id()
                        session.build_chat_completion(
                            CI_prompts["session_start_template"].format(code=file.get(add_line_number=True)),
                        )

                        fix_groups[file_path].append(
                            CodeFixGroup(start_line, end_line, group_errors, session_id, []),
                        )

            # Fix errors in each code block
            with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as progress:
                group_counts = sum([len(groups) for groups in fix_groups.values()])
                task_id = progress.add_task("Fixing repo...", total=group_counts)

                for file_path in fix_groups:
                    file = evo.files[evo.project_path / Path(file_path)]
                    for code_fix_g in fix_groups[file_path]:
                        start_line = code_fix_g.start_line
                        end_line = code_fix_g.end_line
                        group_errors = code_fix_g.errors
                        code_snippet_with_lineno = file.get(
                            start_line,
                            end_line,
                            add_line_number=True,
                            return_list=False,
                        )
                        errors_str = "\n\n".join(str(e) for e in group_errors)

                        # ask LLM to repair current code snippet
                        user_prompt = CI_prompts["session_normal_template"].format(
                            code=code_snippet_with_lineno,
                            lint_info=errors_str,
                            start_line=start_line,
                            end_line=end_line,
                            start_lineno=start_line,
                        )

                        session = api.build_chat_session(conversation_id=code_fix_g.session_id)
                        res = session.build_chat_completion(user_prompt)
                        code_fix_g.responses.append(res)
                        progress.update(
                            task_id,
                            description=f"[green]Fixing[/green] [cyan]{file_path}[/cyan]...",
                            advance=1,
                        )

            # Manual inspection and repair
            for file_path in last_feedback.errors:
                print(
                    Rule(
                        f"[bright_blue]Checking[/bright_blue] [cyan]{file_path}[/cyan]",
                        style="bright_blue",
                        align="left",
                        characters=".",
                    ),
                )

                file = evo.files[evo.project_path / Path(file_path)]

                # generate changes
                for group_id, code_fix_g in enumerate(fix_groups[file_path], start=1):
                    start_line, end_line, group_errors = code_fix_g.start_line, code_fix_g.end_line, code_fix_g.errors
                    session = api.build_chat_session(conversation_id=code_fix_g.session_id)

                    print(f"[yellow]Checking part {group_id}...[/yellow]")

                    front_context = file.get(start_line - 3, start_line - 1)
                    rear_context = file.get(end_line + 1, end_line + 3)
                    front_context_with_lineno = file.get(start_line - 3, start_line - 1, add_line_number=True)
                    rear_context_with_lineno = file.get(end_line + 1, end_line + 3, add_line_number=True)

                    code_snippet_with_lineno = file.get(start_line, end_line, add_line_number=True, return_list=False)

                    # print errors
                    printed_errors_str = "\n".join(
                        [
                            f"[{error.checker}] {error.line: >{file.lineno_width}}:{error.column: <4}"
                            f" {error.code}  {error.msg}"
                            for error in group_errors
                        ],
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

                    res = code_fix_g.responses[0]
                    code_snippet_lines = file.get(start_line, end_line, add_line_number=False, return_list=True)

                    while True:
                        try:
                            new_code = re.search(r".*```[Pp]ython\n(.*?)\n```.*", res, re.DOTALL).group(1)
                        except (re.error, AttributeError) as exc:
                            print(f"[red]Error when extract codes[/red]:\n {res}\nException: {exc}")
                        try:
                            fixed_errors_info = re.search(r".*```[Jj]son\n(.*?)\n```.*", res, re.DOTALL).group(1)
                            fixed_errors_info = json.loads(fixed_errors_info)
                        except AttributeError:
                            fixed_errors_info = None
                        except (json.JSONDecodeError, re.error) as exc:
                            fixed_errors_info = None
                            print(f"[red]Error when extracting fixed_errors[/red]: {exc}")

                        new_code = CodeFile.remove_line_number(new_code)

                        # print repair status (code diff)
                        diff = ndiff(code_snippet_lines, new_code.split("\n"))

                        # add 2 spaces to align with diff format
                        front_context = re.sub(r"^", "  ", front_context, flags=re.MULTILINE)
                        rear_context = re.sub(r"^", "  ", rear_context, flags=re.MULTILINE)

                        table = Table(show_header=False, box=None)
                        table.add_column()
                        table.add_column()
                        table.add_column()
                        table.add_row("", "", Syntax(front_context, lexer="python", background_color="default"))
                        table.add_row("", "", Rule(style="dark_orange"))
                        diff_original_lineno = start_line
                        diff_new_lineno = start_line
                        for i in diff:
                            if i.startswith("+"):
                                table.add_row(
                                    "",
                                    Text(str(diff_new_lineno), style="green bold"),
                                    Text(i, style="green"),
                                )
                                diff_new_lineno += 1
                            elif i.startswith("-"):
                                table.add_row(
                                    Text(str(diff_original_lineno), style="red bold"),
                                    "",
                                    Text(i, style="red"),
                                )
                                diff_original_lineno += 1
                            elif i.startswith("?"):
                                table.add_row("", "", Text(i, style="yellow"))
                            else:
                                table.add_row(
                                    str(diff_original_lineno),
                                    str(diff_new_lineno),
                                    Syntax(i, lexer="python", background_color="default"),
                                )
                                diff_original_lineno += 1
                                diff_new_lineno += 1
                        table.add_row("", "", Rule(style="dark_orange"))
                        table.add_row("", "", Syntax(rear_context, lexer="python", background_color="default"))
                        print(Panel.fit(table, title="Repair Status"))

                        operation = Prompt.ask(
                            "Input your operation [ [red]([bold]s[/bold])kip[/red] / "
                            "[green]([bold]a[/bold])pply[/green] / "
                            "[yellow]manual instruction[/yellow] ]",
                        )
                        print()
                        if operation in ("s", "skip"):
                            fix_records[file_path].skipped_errors.extend(group_errors)
                            break
                        if operation in ("a", "apply"):
                            if fixed_errors_info:
                                fixed_errors_str = "\n".join(fixed_errors_info["errors"])
                                for error in group_errors:
                                    if f"{error.line}:{error.column}" in fixed_errors_str:
                                        fix_records[file_path].manually_fixed_errors.append(error)
                                    else:
                                        fix_records[file_path].skipped_errors.append(error)
                            else:
                                fix_records[file_path].directly_fixed_errors.extend(group_errors)

                            changes[file_path].append((start_line, end_line, new_code))
                            break

                        fix_records[file_path].manual_instructions[operation].extend(group_errors)
                        res = session.build_chat_completion(
                            CI_prompts["session_manual_template"].format(operation=operation),
                        )
                        code_fix_g.responses.append(res)

                # apply changes
                file.apply_changes(changes[file_path])

            evo.fix_records = fix_records

        return evo


class CIEvoAgent(EvoAgent):
    def __init__(self, evolving_strategy: CIEvoStr) -> None:
        super().__init__(max_loop=1, evolving_strategy=evolving_strategy)
        self.evolving_trace = []

    def multistep_evolve(self, evo: Repo, eva: Evaluator) -> Repo:
        evo = self.evolving_strategy.evolve(
            evo=evo,
            evolving_trace=self.evolving_trace,
        )

        self.evolving_trace.append(EvoStep(evo, feedback=eva.evaluate(evo)))

        return evo


DIR = None
while DIR is None or not DIR.exists():
    DIR = Prompt.ask("Please input the [cyan]project directory[/cyan]")
    DIR = Path(DIR)

excludes = Prompt.ask(
    "Input the [dark_orange]excluded directories[/dark_orange] (relative to "
    "[cyan]project path[/cyan] and separated by whitespace)",
).split(" ")
excludes = [Path(exclude.strip()) for exclude in excludes if exclude.strip() != ""]

start_time = time.time()
start_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%m%d%H%M")

repo = Repo(DIR, excludes=excludes)
# evaluator = MultiEvaluator(MypyEvaluator(), RuffEvaluator())
evaluator = RuffEvaluator()
estr = CIEvoStr()
ea = CIEvoAgent(estr)
ea.multistep_evolve(repo, evaluator)
while True:
    print(Rule(f"Round {len(ea.evolving_trace)} repair", style="blue"))
    repo: Repo = ea.multistep_evolve(repo, evaluator)

    fix_records = repo.fix_records
    filename = f"{DIR.name}_{start_timestamp}_round_{len(ea.evolving_trace)}_fix_records.json"
    with Path(filename).open("w") as file:
        json.dump({k: v.to_dict() for k, v in fix_records.items()}, file, indent=4)

    # Count the number of skipped errors
    skipped_errors_count = 0
    directly_fixed_errors_count = 0
    manually_fixed_errors_count = 0
    skipped_errors_code_count = defaultdict(int)
    directly_fixed_errors_code_count = defaultdict(int)
    manually_fixed_errors_code_count = defaultdict(int)
    code_message = defaultdict(str)
    for record in fix_records.values():
        skipped_errors_count += len(record.skipped_errors)
        directly_fixed_errors_count += len(record.directly_fixed_errors)
        manually_fixed_errors_count += len(record.manually_fixed_errors)
        for error in record.skipped_errors:
            skipped_errors_code_count[error.code] += 1
            code_message[error.code] = error.msg
        for error in record.directly_fixed_errors:
            directly_fixed_errors_code_count[error.code] += 1
            code_message[error.code] = error.msg
        for error in record.manually_fixed_errors:
            manually_fixed_errors_code_count[error.code] += 1
            code_message[error.code] = error.msg

    skipped_errors_statistics = ""
    directly_fixed_errors_statistics = ""
    manually_fixed_errors_statistics = ""
    for code, count in sorted(skipped_errors_code_count.items(), key=lambda x: x[1], reverse=True):
        skipped_errors_statistics += f"{count: >5} {code: >10} {code_message[code]}\n"
    for code, count in sorted(directly_fixed_errors_code_count.items(), key=lambda x: x[1], reverse=True):
        directly_fixed_errors_statistics += f"{count: >5} {code: >10} {code_message[code]}\n"
    for code, count in sorted(manually_fixed_errors_code_count.items(), key=lambda x: x[1], reverse=True):
        manually_fixed_errors_statistics += f"{count: >5} {code: >10} {code_message[code]}\n"

    # Create a table to display the counts and ratios
    table = Table(title="Error Fix Statistics")
    table.add_column("Type")
    table.add_column("Statistics")
    table.add_column("Count")
    table.add_column("Ratio")

    total_errors_count = skipped_errors_count + directly_fixed_errors_count + manually_fixed_errors_count
    table.add_row("Total Errors", "", Text(str(total_errors_count), style="cyan"), "")
    table.add_row(
        Text("Skipped Errors", style="red"),
        skipped_errors_statistics,
        Text(str(skipped_errors_count), style="red"),
        Text(f"{skipped_errors_count / total_errors_count:.2%}"),
        style="red",
    )
    table.add_row(
        Text("Directly Fixed Errors", style="green"),
        directly_fixed_errors_statistics,
        Text(str(directly_fixed_errors_count), style="green"),
        Text(f"{directly_fixed_errors_count / total_errors_count:.2%}"),
        style="green",
    )
    table.add_row(
        Text("Manually Fixed Errors", style="yellow"),
        manually_fixed_errors_statistics,
        Text(str(manually_fixed_errors_count), style="yellow"),
        Text(f"{manually_fixed_errors_count / total_errors_count:.2%}"),
        style="yellow",
    )

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
