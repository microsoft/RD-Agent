import unittest
from pathlib import Path

import nbformat
from rich import print

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.experiment.workspace import KGFBWorkspace
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T


class TestTpl(unittest.TestCase):
    def test_competition_template(self):
        competition = KAGGLE_IMPLEMENT_SETTING.competition
        print(f"[bold orange]{competition}[/bold orange]")
        ws = KGFBWorkspace(
            template_folder_path=Path(__file__).parent.parent.parent
            / "rdagent/scenarios/kaggle/experiment"
            / f"{KAGGLE_IMPLEMENT_SETTING.competition}_template"
        )
        print(ws.workspace_path)
        ws.execute()
        success = (ws.workspace_path / "submission.csv").exists()
        ws.clear()
        return success


if __name__ == "__main__":
    unittest.main()
