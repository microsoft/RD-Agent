from __future__ import annotations

import json
import mimetypes
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class UISupabaseSettings(ExtendedBaseSettings):
    model_config = SettingsConfigDict(env_prefix="UI_SUPABASE_", protected_namespaces=())

    enabled: bool = False
    url: str = ""
    service_role_key: str = ""
    bucket: str = "rdagent"
    trace_prefix: str = "traces"
    stdout_prefix: str = "stdout"
    upload_prefix: str = "uploads"
    request_timeout: int = 30

    def is_enabled(self) -> bool:
        return self.enabled and bool(self.url and self.service_role_key and self.bucket)


UI_SUPABASE_SETTINGS = UISupabaseSettings()


def _join_object_path(*parts: str) -> str:
    normalized_parts = [part.strip("/") for part in parts if part and part.strip("/")]
    return str(PurePosixPath(*normalized_parts))


def trace_object_prefix(trace_id: str, settings: UISupabaseSettings = UI_SUPABASE_SETTINGS) -> str:
    return _join_object_path(settings.trace_prefix, trace_id)


def stdout_object_path(trace_id: str, settings: UISupabaseSettings = UI_SUPABASE_SETTINGS) -> str:
    return _join_object_path(settings.stdout_prefix, f"{trace_id}.log")


def upload_object_path(
    scenario: str,
    trace_name: str,
    filename: str,
    settings: UISupabaseSettings = UI_SUPABASE_SETTINGS,
) -> str:
    return _join_object_path(settings.upload_prefix, scenario, trace_name, filename)


class SupabaseStorageClient:
    def __init__(self, settings: UISupabaseSettings = UI_SUPABASE_SETTINGS) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        return self.settings.is_enabled()

    def upload_bytes(
        self,
        payload: bytes,
        object_path: str,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        response = requests.post(
            self._object_url(object_path),
            headers=self._headers(content_type=content_type),
            data=payload,
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()

    def upload_file(self, local_path: str | Path, object_path: str) -> None:
        path = Path(local_path)
        content_type, _ = mimetypes.guess_type(str(path))
        self.upload_bytes(
            path.read_bytes(),
            object_path,
            content_type=content_type or "application/octet-stream",
        )

    def upload_directory(self, local_dir: str | Path, object_prefix: str) -> list[str]:
        root = Path(local_dir)
        uploaded_files: list[str] = []

        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            rel_path = file_path.relative_to(root).as_posix()
            self.upload_file(file_path, _join_object_path(object_prefix, rel_path))
            uploaded_files.append(rel_path)

        self.upload_bytes(
            json.dumps(uploaded_files, indent=2).encode("utf-8"),
            _join_object_path(object_prefix, "__manifest__.json"),
            content_type="application/json",
        )
        return uploaded_files

    def download_file(self, object_path: str, local_path: str | Path) -> Path:
        response = requests.get(
            self._object_url(object_path),
            headers=self._headers(),
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()

        target_path = Path(local_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(response.content)
        return target_path

    def download_directory(self, object_prefix: str, local_dir: str | Path) -> list[Path]:
        local_root = Path(local_dir)
        manifest_path = local_root / "__manifest__.json"
        self.download_file(_join_object_path(object_prefix, "__manifest__.json"), manifest_path)

        uploaded_files = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(uploaded_files, list):
            error_message = f"Invalid manifest for {object_prefix}: expected list, got {type(uploaded_files)!r}"
            raise ValueError(error_message)

        downloaded_paths: list[Path] = []
        for rel_path in uploaded_files:
            if not isinstance(rel_path, str):
                error_message = f"Invalid manifest entry for {object_prefix}: expected str, got {type(rel_path)!r}"
                raise ValueError(error_message)
            downloaded_paths.append(self.download_file(_join_object_path(object_prefix, rel_path), local_root / rel_path))

        return downloaded_paths

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.settings.service_role_key,
            "Authorization": f"Bearer {self.settings.service_role_key}",
            "x-upsert": "true",
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        return headers

    def _object_url(self, object_path: str) -> str:
        object_key = quote(object_path, safe="/")
        return f"{self.settings.url.rstrip('/')}/storage/v1/object/{self.settings.bucket}/{object_key}"
