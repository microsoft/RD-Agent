import os
import tempfile
import unittest
from pathlib import Path

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
        # create a symbolic link inside workspace and ensure checkpoint preserves the link
        external_file = self.tmp_path / "external.txt"
        external_file.write_text("external data")
        ws = FBWorkspace()
        ws.workspace_path = self.tmp_path / "ws"
        ws.prepare()
        (ws.workspace_path / "sym.txt").symlink_to(external_file)
        ws.inject_files(**{"foo.py": "print('hi')", "bar.py": "x = 1"})

        # Snapshot current workspace
        original_files = {
            p.relative_to(ws.workspace_path): (os.readlink(p) if p.is_symlink() else p.read_text())
            for p in ws.workspace_path.rglob("*")
            if p.is_file() or p.is_symlink()
        }
        ws.create_ws_ckp()
        self.assertIsNotNone(ws.ws_ckp, "Checkpoint data should have been generated")

        # Mutate workspace
        (ws.workspace_path / "foo.py").write_text("print('changed')")
        (ws.workspace_path / "new.py").write_text("pass")
        (ws.workspace_path / "sym.txt").unlink()

        # Restore and verify equality with snapshot
        ws.recover_ws_ckp()

        # Ensure symbolic link still exists after recovery.
        self.assertTrue((ws.workspace_path / "sym.txt").is_symlink())
        recovered_files = {
            p.relative_to(ws.workspace_path): (os.readlink(p) if p.is_symlink() else p.read_text())
            for p in ws.workspace_path.rglob("*")
            if p.is_file() or p.is_symlink()
        }
        self.assertEqual(recovered_files, original_files)

        # Verify large files (>100 KB) are excluded when a size-limit is configured.
        from rdagent.core.conf import RD_AGENT_SETTINGS as _SETTINGS

        _SETTINGS.workspace_ckp_size_limit = 100 * 1024  # set limit temporarily for this test

        large_file = ws.workspace_path / "large.bin"
        large_file.write_bytes(b"0" * (110 * 1024))  # 110 KB dummy content
        ws.create_ws_ckp()
        ws.recover_ws_ckp()
        self.assertFalse((ws.workspace_path / "large.bin").exists())
