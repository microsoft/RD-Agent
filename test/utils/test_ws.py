from pathlib import Path

import unittest
import tempfile

from rdagent.core.experiment import FBWorkspace


class TestFBWorkspace(unittest.TestCase):
    """
    Unit-tests for `FBWorkspace`.
    """

    def setUp(self) -> None:  # noqa: D401
        """
        Create an isolated temporary directory for each test case.
        """
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp_dir.name)

    def tearDown(self) -> None:
        """
        Clean up the temporary directory created in :py:meth:`setUp`.
        """
        self._tmp_dir.cleanup()

    def test_checkpoint_roundtrip(self) -> None:
        """
        Verify that ``create_ws_ckp`` captures the current workspace state and
        ``recover_ws_ckp`` faithfully restores it.
        """
        ws = FBWorkspace()
        ws.workspace_path = self.tmp_path / "ws"
        ws.inject_files(**{"foo.py": "print('hi')", "bar.py": "x = 1"})

        # Snapshot current workspace
        original_files = {
            p.relative_to(ws.workspace_path): p.read_text()
            for p in ws.workspace_path.rglob("*")
            if p.is_file()
        }
        ws.create_ws_ckp()
        self.assertIsNotNone(ws.ws_ckp, "Checkpoint data should have been generated")

        # Mutate workspace
        (ws.workspace_path / "foo.py").write_text("print('changed')")
        (ws.workspace_path / "new.py").write_text("pass")

        # Restore and verify equality with snapshot
        ws.recover_ws_ckp()
        recovered_files = {
            p.relative_to(ws.workspace_path): p.read_text()
            for p in ws.workspace_path.rglob("*")
            if p.is_file()
        }
        self.assertEqual(recovered_files, original_files)
