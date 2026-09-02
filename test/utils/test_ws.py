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

    def test_workspace_file_operations_reject_path_escape(self) -> None:
        ws = FBWorkspace()
        ws.workspace_path = self.tmp_path / "ws"
        ws.prepare()

        ws.inject_files(**{"file.py": "a", "nested/file.py": "b"})
        self.assertEqual((ws.workspace_path / "file.py").read_text(), "a")
        self.assertEqual((ws.workspace_path / "nested/file.py").read_text(), "b")

        outside = self.tmp_path / "outside.txt"
        outside.write_text("keep")
        absolute_outside = self.tmp_path / "absolute.txt"
        absolute_inside = ws.workspace_path / "absolute.txt"

        for escaped_path in ("../outside.txt", "nested/../file.py", str(absolute_outside), str(absolute_inside)):
            with self.subTest(operation="inject", path=escaped_path), self.assertRaises(ValueError):
                ws.inject_files(**{escaped_path: "escaped"})
            with self.subTest(operation="inject-delete", path=escaped_path), self.assertRaises(ValueError):
                ws.inject_files(**{escaped_path: FBWorkspace.DEL_KEY})
            with self.subTest(operation="remove", path=escaped_path), self.assertRaises(ValueError):
                ws.remove_files(escaped_path)

        self.assertEqual(outside.read_text(), "keep")
        self.assertFalse(absolute_outside.exists())

    def test_workspace_file_operations_reject_symlink_escape(self) -> None:
        ws = FBWorkspace()
        ws.workspace_path = self.tmp_path / "ws"
        ws.prepare()

        outside = self.tmp_path / "outside"
        outside.mkdir()
        (ws.workspace_path / "link").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(ValueError):
            ws.inject_files(**{"link/escaped.txt": "escaped"})

        outside_file = outside / "existing.txt"
        outside_file.write_text("keep")
        with self.assertRaises(ValueError):
            ws.remove_files("link/existing.txt")

        self.assertFalse((outside / "escaped.txt").exists())
        self.assertEqual(outside_file.read_text(), "keep")

    def test_workspace_delete_unlinks_leaf_symlink_without_deleting_target(self) -> None:
        ws = FBWorkspace()
        ws.workspace_path = self.tmp_path / "ws"
        ws.prepare()

        target = ws.workspace_path / "target.txt"
        target.write_text("keep")
        link = ws.workspace_path / "link.txt"
        link.symlink_to(target)

        ws.remove_files("link.txt")

        self.assertFalse(link.is_symlink())
        self.assertEqual(target.read_text(), "keep")

        outside_target = self.tmp_path / "outside-target.txt"
        outside_target.write_text("keep")
        outside_link = ws.workspace_path / "outside-link.txt"
        outside_link.symlink_to(outside_target)
        outside_target.unlink()

        ws.inject_files(**{"outside-link.txt": FBWorkspace.DEL_KEY})

        self.assertFalse(outside_link.is_symlink())
        self.assertFalse(outside_target.exists())
