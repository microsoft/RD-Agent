"""
Developers concentrating on writing documents for a workspace
"""

from rdagent.core.developer import Developer
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.ret import MarkdownAgentOut
from rdagent.utils.agent.tpl import T


class DocDev(Developer[Experiment]):
    """
    The developer is responsible for writing documents for a workspace.
    """

    def develop(self, exp: Experiment) -> None:
        """
        Write documents for the workspace.
        """
        ws: FBWorkspace = exp.experiment_workspace

        file_li = [str(file.relative_to(ws.workspace_path)) for file in ws.workspace_path.rglob("*") if file.is_file()]

        key_file_list = ["main.py", "scores.csv"]

        system_prompt = T(".prompts:docdev.system").r()
        user_prompt = T(".prompts:docdev.user").r(
            file_li=file_li,
            key_files={f: (ws.workspace_path / f).read_text() for f in key_file_list},
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt
        )
        markdown = MarkdownAgentOut.extract_output(resp)
        ws.inject_files(**{"README.md": markdown})
