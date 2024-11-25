import unittest
from pathlib import Path

import nbformat
from rich import print

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.experiment.workspace import KGFBWorkspace
from rdagent.scenarios.kaggle.kaggle_crawler import download_data
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T


class TestTpl(unittest.TestCase):
    def test_competition_template(self):
        """
        export KG_COMPETITION=<competition_name> before running this test
        """
        competition = KAGGLE_IMPLEMENT_SETTING.competition
        print(f"[bold orange]{competition}[/bold orange]")
        download_data(competition)
        ws = KGFBWorkspace(
            template_folder_path=Path(__file__).parent.parent.parent
            / KAGGLE_IMPLEMENT_SETTING.template_path
            / f"{competition}",
        )
        print(ws.workspace_path)
        ws.execute()
        success = (ws.workspace_path / "submission.csv").exists()
        self.assertTrue(success, "submission.csv is not generated")
        # ws.clear()


if __name__ == "__main__":
    unittest.main()
