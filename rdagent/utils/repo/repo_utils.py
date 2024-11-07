import ast
import inspect
import os
from pathlib import Path
from typing import Dict, List, Union


class RepoAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.summaries = {}

    def summarize_repo(self, verbose_level: int = 1, doc_str_level: int = 1, sign_level: int = 1) -> str:
        """
        Generate a natural language summary of the entire repository workspace.

        :param verbose_level: Level of verbosity for the summary (0-2)
        :param doc_str_level: Level of detail for docstrings (0-2)
        :param sign_level: Level of detail for function signatures (0-2)
        :return: A string containing the workspace summary
        """
        file_summaries = []
        tree_structure = self._generate_tree_structure()

        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.repo_path)
                    file_summaries.append(self._summarize_file(file_path, verbose_level, doc_str_level, sign_level))

        total_files = len(file_summaries)
        workspace_summary = f"Workspace Summary for {self.repo_path.name}\n"
        workspace_summary += f"{'=' * 40}\n\n"
        workspace_summary += "Workspace Structure:\n"
        workspace_summary += tree_structure
        workspace_summary += (
            f"\nThis workspace contains {total_files} Python file{'s' if total_files != 1 else ''}.\n\n"
        )

        for i, summary in enumerate(file_summaries, 1):
            workspace_summary += f"File {i} of {total_files}:\n{summary}\n"

        workspace_summary += f"\nEnd of Workspace Summary for {self.repo_path.name}"
        return workspace_summary

    def _generate_tree_structure(self) -> str:
        """
        Generate a tree-like structure of the repository.
        """
        tree = []
        for root, dirs, files in os.walk(self.repo_path):
            level = root.replace(str(self.repo_path), "").count(os.sep)
            indent = "│   " * (level - 1) + "├── " if level > 0 else ""
            rel_path = os.path.relpath(root, self.repo_path)
            tree.append(f"{indent}{os.path.basename(root)}/")

            subindent = "│   " * level + "├── "
            for file in files:
                if file.endswith(".py"):
                    tree.append(f"{subindent}{file}")

        return "\n".join(tree)

    def _summarize_file(self, file_path: Path, verbose_level: int, doc_str_level: int, sign_level: int) -> str:
        with open(file_path, "r") as f:
            content = f.read()

        tree = ast.parse(content)
        summary = f"File: {file_path.relative_to(self.repo_path)}\n"
        summary += f"{'-' * 40}\n"

        classes = [node for node in ast.iter_child_nodes(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.iter_child_nodes(tree) if isinstance(node, ast.FunctionDef)]

        if classes:
            summary += f"This file contains {len(classes)} class{'es' if len(classes) > 1 else ''}.\n"
        if functions:
            summary += f"This file contains {len(functions)} top-level function{'s' if len(functions) > 1 else ''}.\n"

        for node in classes + functions:
            if isinstance(node, ast.ClassDef):
                summary += self._summarize_class(node, verbose_level, doc_str_level, sign_level)
            elif isinstance(node, ast.FunctionDef):
                summary += self._summarize_function(node, verbose_level, doc_str_level, sign_level)

        return summary

    def _summarize_class(self, node: ast.ClassDef, verbose_level: int, doc_str_level: int, sign_level: int) -> str:
        summary = f"\nClass: {node.name}\n"
        if doc_str_level > 0 and ast.get_docstring(node):
            summary += f"  Description: {ast.get_docstring(node).split('.')[0]}.\n"

        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if methods:
            summary += f"  This class has {len(methods)} method{'s' if len(methods) > 1 else ''}.\n"

        if verbose_level > 1:
            for method in methods:
                summary += self._summarize_function(method, verbose_level, doc_str_level, sign_level, indent="  ")
        return summary

    def _summarize_function(
        self, node: ast.FunctionDef, verbose_level: int, doc_str_level: int, sign_level: int, indent: str = ""
    ) -> str:
        summary = f"{indent}Function: {node.name}\n"
        if sign_level > 0:
            # Generate the function signature
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")

            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            signature = f"{node.name}({', '.join(args)}){returns}"
            summary += f"{indent}  Signature: {signature}\n"

        if doc_str_level > 0 and ast.get_docstring(node):
            doc = ast.get_docstring(node)
            summary += f"{indent}  Purpose: {doc.split('.')[0]}.\n"
        return summary

    def highlight(self, file_names: Union[str, List[str]]) -> Dict[str, str]:
        """
        Extract content from specified file(s) within the repo.

        :param file_names: A single file name or a list of file names to highlight
        :return: Dictionary of file names and their content
        """
        if isinstance(file_names, str):
            file_names = [file_names]

        highlighted_content = {}
        for file_name in file_names:
            file_path = self.repo_path / file_name
            if file_path.exists() and file_path.is_file():
                with open(file_path, "r") as f:
                    highlighted_content[file_name] = f.read()
            else:
                highlighted_content[file_name] = f"File not found: {file_name}"

        return highlighted_content


if __name__ == "__main__":
    analyzer = RepoAnalyzer(repo_path="features")
    summary = analyzer.summarize_repo(verbose_level=2, doc_str_level=2, sign_level=2)
    print(summary)
    highlighted_files = analyzer.highlight(
        file_names=["utils/repo/repo_utils.py", "components/benchmark/eval_method.py"]
    )
    print("\nHighlighted Files:")
    for file_name, content in highlighted_files.items():
        print(f"\n{file_name}\n{'=' * len(file_name)}\n{content}")
