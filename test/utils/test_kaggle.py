import unittest
from pathlib import Path

from rich import print

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.scenarios.kaggle.experiment.workspace import KGFBWorkspace
from rdagent.scenarios.kaggle.kaggle_crawler import download_data


class TestTpl(unittest.TestCase):
    def test_competition_template(self):
        """
        export KG_COMPETITION=<competition_name> before running this test
        """
        competition = KAGGLE_IMPLEMENT_SETTING.competition
        print(f"[bold orange]{competition}[/bold orange]")
        download_data(competition, settings=KAGGLE_IMPLEMENT_SETTING)
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
