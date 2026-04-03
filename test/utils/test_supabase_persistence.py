import json
import tempfile
import unittest
from pathlib import Path

import pytest

from rdagent.log.server.persistence import (
    SupabaseStorageClient,
    UISupabaseSettings,
    stdout_object_path,
    trace_object_prefix,
    upload_object_path,
)


class DummySupabaseClient(SupabaseStorageClient):
    def __init__(self) -> None:
        super().__init__(
            UISupabaseSettings(
                enabled=True,
                url="https://example.supabase.co",
                service_role_key="service-role-key",
                bucket="rdagent-artifacts",
            )
        )
        self.remote_payloads: dict[str, bytes] = {}

    def upload_bytes(
        self,
        payload: bytes,
        object_path: str,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.remote_payloads[object_path] = payload

    def download_file(self, object_path: str, local_path: str | Path) -> Path:
        target_path = Path(local_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(self.remote_payloads[object_path])
        return target_path


@pytest.mark.offline
class SupabasePersistenceTest(unittest.TestCase):
    def test_object_path_helpers(self) -> None:
        settings = UISupabaseSettings(
            enabled=True,
            url="https://example.supabase.co",
            service_role_key="service-role-key",
            bucket="rdagent-artifacts",
            trace_prefix="trace-root",
            stdout_prefix="stdout-root",
            upload_prefix="upload-root",
        )

        self.assertEqual(trace_object_prefix("Data Science/demo", settings), "trace-root/Data Science/demo")
        self.assertEqual(stdout_object_path("Data Science/demo", settings), "stdout-root/Data Science/demo.log")
        self.assertEqual(
            upload_object_path("Data Science", "demo", "report.pdf", settings),
            "upload-root/Data Science/demo/report.pdf",
        )

    def test_upload_directory_writes_manifest(self) -> None:
        client = DummySupabaseClient()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "nested").mkdir()
            (root / "summary.txt").write_text("summary", encoding="utf-8")
            (root / "nested" / "details.json").write_text('{"ok": true}', encoding="utf-8")

            uploaded_files = client.upload_directory(root, "traces/demo")

        self.assertEqual(uploaded_files, ["nested/details.json", "summary.txt"])
        manifest = json.loads(client.remote_payloads["traces/demo/__manifest__.json"].decode("utf-8"))
        self.assertEqual(manifest, ["nested/details.json", "summary.txt"])
        self.assertEqual(client.remote_payloads["traces/demo/summary.txt"], b"summary")

    def test_download_directory_uses_manifest(self) -> None:
        client = DummySupabaseClient()
        client.remote_payloads["traces/demo/__manifest__.json"] = json.dumps(
            ["nested/details.json", "summary.txt"]
        ).encode("utf-8")
        client.remote_payloads["traces/demo/nested/details.json"] = b'{"ok": true}'
        client.remote_payloads["traces/demo/summary.txt"] = b"summary"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            downloaded_paths = client.download_directory("traces/demo", root)

            self.assertEqual(
                sorted(path.relative_to(root).as_posix() for path in downloaded_paths),
                ["nested/details.json", "summary.txt"],
            )
            self.assertEqual((root / "summary.txt").read_text(encoding="utf-8"), "summary")


if __name__ == "__main__":
    unittest.main()
