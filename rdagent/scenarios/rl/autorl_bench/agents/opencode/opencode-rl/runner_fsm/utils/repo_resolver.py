from __future__ import annotations

import json
import os
import re
import subprocess
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GITHUB_ARCHIVE_HOSTS = {"github.com", "www.github.com"}
_HF_HOSTS = {"huggingface.co", "www.huggingface.co"}
_PREFERRED_CLONES_BASE = Path(
    os.environ.get("OPENCODE_FSM_CLONES_DIR") or "/data/tiansha/opencode_fsm_targets"
)

def is_probably_repo_url(repo: str) -> bool:
    s = str(repo or "").strip()
    if not s:
        return False
    if s.startswith(("http://", "https://", "ssh://", "git@")):
        return True
    if s.endswith(".git") and ("/" in s or ":" in s):
        return True
    if _OWNER_REPO_RE.match(s):
        return True
    return False

def normalize_repo_url(repo: str) -> str:
    """Normalize shorthand forms into a git-cloneable URL."""
    s = str(repo or "").strip()
    if _OWNER_REPO_RE.match(s):
        # Default to GitHub for shorthand `owner/repo`.
        return f"https://github.com/{s}.git"
    return s

def _download_file(
    url: str,
    *,
    out_path: Path,
    timeout_seconds: int = 60,
    headers: dict[str, str] | None = None,
    max_bytes: int | None = None,
) -> tuple[bool, str]:
    try:
        h = {
            "User-Agent": "opencode-fsm/1.0",
            "Accept": "application/octet-stream",
        }
        if headers:
            h.update({str(k): str(v) for k, v in headers.items() if str(k).strip()})
        req = Request(
            url,
            headers=h,
            method="GET",
        )
        with urlopen(req, timeout=timeout_seconds) as resp:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                total = 0
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes is not None and total > int(max_bytes):
                        raise OSError(f"max_bytes_exceeded: {max_bytes}")
                    f.write(chunk)
        return True, ""
    except HTTPError as e:
        return False, f"HTTPError {getattr(e, 'code', '')}: {str(e)}"
    except URLError as e:
        return False, f"URLError: {str(e)}"
    except OSError as e:
        return False, f"OSError: {str(e)}"

def _download_hf_dataset_snapshot(
    *,
    namespace: str,
    name: str,
    dest: Path,
    env: dict[str, str],
) -> tuple[bool, str]:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    token = (
        str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or "")
        .strip()
        or None
    )
    try:
        url = f"https://huggingface.co/api/datasets/{namespace}/{name}"
        headers: dict[str, str] = {"Accept": "application/json", "User-Agent": "opencode-fsm/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=20) as resp:
            raw = resp.read()
        info = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(info, dict):
            raise RuntimeError("hf_api_invalid_json")
    except HTTPError as e:
        return False, f"hf_api_http_error {getattr(e, 'code', '')}: {e}"
    except URLError as e:
        return False, f"hf_api_url_error: {e}"
    except Exception as e:
        return False, f"hf_api_error: {e}"

    gated = bool(info.get("gated") or False)
    private = bool(info.get("private") or False)
    if (gated or private) and not token:
        return False, "hf_dataset_requires_token (set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN)"

    revision = str(info.get("sha") or "main").strip() or "main"
    siblings = info.get("siblings") or []
    if not isinstance(siblings, list):
        siblings = []

    max_total_bytes = int(os.environ.get("OPENCODE_FSM_HF_MAX_TOTAL_BYTES") or 512 * 1024 * 1024)
    max_file_bytes = int(os.environ.get("OPENCODE_FSM_HF_MAX_FILE_BYTES") or 256 * 1024 * 1024)
    total_bytes = 0
    downloaded: list[dict[str, object]] = []
    errors: list[str] = []

    extra_headers: dict[str, str] = {}
    if token:
        extra_headers["Authorization"] = f"Bearer {token}"

    for s in siblings:
        if not isinstance(s, dict):
            continue
        rfilename = str(s.get("rfilename") or "").strip()
        if not rfilename:
            continue
        # Skip very unlikely problematic paths.
        if rfilename.startswith(("/", "\\")) or ".." in rfilename.split("/"):
            errors.append(f"skip_unsafe_path: {rfilename}")
            continue

        if total_bytes >= max_total_bytes:
            errors.append(f"max_total_bytes_exceeded: {max_total_bytes}")
            break

        encoded = quote(rfilename, safe="/")
        url = f"https://huggingface.co/datasets/{namespace}/{name}/resolve/{revision}/{encoded}"
        out_path = dest / rfilename
        ok, err = _download_file(
            url,
            out_path=out_path,
            timeout_seconds=120,
            headers=extra_headers,
            max_bytes=min(max_file_bytes, max_total_bytes - total_bytes),
        )
        if not ok:
            errors.append(f"{rfilename}: {err}")
            continue
        try:
            size = out_path.stat().st_size
        except OSError:
            size = None
        if isinstance(size, int):
            total_bytes += int(size)
        downloaded.append({"path": rfilename, "bytes": size})

    manifest = {
        "source": "huggingface_dataset",
        "dataset_id": f"{namespace}/{name}",
        "revision": revision,
        "gated": gated,
        "private": private,
        "downloaded_files": downloaded,
        "total_bytes": total_bytes,
        "errors": errors,
    }
    (dest / ".opencode_fsm").mkdir(exist_ok=True)
    (dest / "data").mkdir(exist_ok=True)
    (dest / "data" / "hf_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Initialize git for revert guards (best effort).
    subprocess.run(["git", "-C", str(dest), "init"], env=env, check=False, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "opencode-fsm"], env=env, check=False)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "opencode-fsm@example.com"], env=env, check=False)
    subprocess.run(["git", "-C", str(dest), "add", "-A"], env=env, check=False, capture_output=True, text=True)
    subprocess.run(
        ["git", "-C", str(dest), "commit", "-m", "init hf snapshot", "--no-gpg-sign"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    if errors and not downloaded:
        return False, "hf_download_failed: " + "; ".join(errors[-5:])
    return True, ""

def _archive_clone_github(
    *,
    owner: str,
    repo: str,
    dest: Path,
    env: dict[str, str],
    timeout_seconds: int = 60,
) -> tuple[bool, str]:
    """Best-effort fallback clone via GitHub archive zip.

    Returns (ok, detail). On success, the repo is extracted into dest.
    """
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    zip_path = dest.parent / f"{dest.name}.zip"
    extract_dir = dest.parent / f"{dest.name}_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    try:
        for branch in ("main", "master"):
            url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
            ok, err = _download_file(url, out_path=zip_path, timeout_seconds=timeout_seconds)
            if not ok:
                errors.append(f"{url}: {err}")
                continue

            try:
                extract_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(extract_dir)
                except zipfile.BadZipFile as e:
                    raise RuntimeError(f"invalid_zip: {e}") from e

                dirs = [p for p in extract_dir.iterdir() if p.is_dir()]
                if len(dirs) == 1:
                    root = dirs[0]
                else:
                    prefix = f"{repo}-"
                    candidates2 = [d for d in dirs if d.name.startswith(prefix)]
                    if len(candidates2) == 1:
                        root = candidates2[0]
                    else:
                        raise RuntimeError(f"unexpected_zip_layout: dirs={[d.name for d in dirs]}")
            except Exception as e:
                errors.append(f"{url}: extract_failed: {e}")
                continue

            try:
                shutil.move(str(root), str(dest))
            except Exception as e:
                errors.append(f"{url}: move_failed: {e}")
                continue

            # Initialize git for revert guards (best effort).
            subprocess.run(["git", "-C", str(dest), "init"], env=env, check=False, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(dest), "config", "user.name", "opencode-fsm"], env=env, check=False)
            subprocess.run(["git", "-C", str(dest), "config", "user.email", "opencode-fsm@example.com"], env=env, check=False)
            subprocess.run(["git", "-C", str(dest), "add", "-A"], env=env, check=False, capture_output=True, text=True)
            subprocess.run(
                ["git", "-C", str(dest), "commit", "-m", "init", "--no-gpg-sign"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            return True, f"archive_branch={branch}"
    finally:
        try:
            zip_path.unlink()
        except Exception:
            pass
        shutil.rmtree(extract_dir, ignore_errors=True)

    return False, "; ".join(errors[-5:])  # tail

@dataclass(frozen=True)
class PreparedRepo:

    repo: Path
    cloned_from: str | None = None

def _find_reusable_clone(base: Path, *, prefix: str) -> Path | None:
    """Best-effort: reuse an existing clone/snapshot under `base` when available.

    Motivation: when callers pass an explicit `clones_dir`, they usually want stable
    caching to avoid repeated network downloads and repeated OpenCode scaffolding.
    We keep the behavior generic: no repo-specific knowledge, only directory naming
    conventions produced by this file.
    """
    base = Path(base).expanduser().resolve()
    pref = str(prefix or "").strip()
    if not pref:
        return None

    candidates: list[Path] = []
    stable = (base / pref).resolve()
    if stable.exists():
        candidates.append(stable)
    candidates.extend(sorted(base.glob(f"{pref}_*")))

    reusable: list[Path] = []
    for p in candidates:
        try:
            if not p.exists() or not p.is_dir():
                continue
        except Exception:
            continue
        # Marker(s) that strongly suggest this directory is a complete fetched snapshot.
        try:
            if (p / ".git").exists() or (p / "data" / "hf_manifest.json").exists():
                reusable.append(p)
        except Exception:
            continue

    reusable_with_mtime: list[tuple[float, Path]] = []
    for p in reusable:
        try:
            mt = float(p.stat().st_mtime)
        except Exception:
            mt = 0.0
        reusable_with_mtime.append((mt, p))
    reusable_with_mtime.sort(key=lambda t: t[0], reverse=True)
    return reusable_with_mtime[0][1].resolve() if reusable_with_mtime else None

def prepare_repo(repo_arg: str, *, clones_dir: Path | None = None) -> PreparedRepo:
    raw = str(repo_arg or "").strip()
    if not raw:
        raise ValueError("--repo is required")

    p = Path(raw).expanduser()
    if p.exists():
        return PreparedRepo(repo=p.resolve(), cloned_from=None)

    if not is_probably_repo_url(raw):
        raise FileNotFoundError(f"repo path not found: {raw}")

    hf = None
    u = str(raw or "").strip()
    if u.startswith(("http://", "https://")):
        parsed = urlparse(u)
        host = (parsed.hostname or "").strip().lower()
        if host in _HF_HOSTS:
            parts = [p for p in (parsed.path or "").split("/") if p]
            if len(parts) >= 3 and parts[0] == "datasets":
                namespace = parts[1].strip()
                name = parts[2].strip()
                if namespace and name:
                    hf = (namespace, name)

    base = clones_dir
    if base is None:
        candidates = [
            _PREFERRED_CLONES_BASE,
            Path(tempfile.gettempdir()) / "opencode_fsm_targets",
        ]
        base2 = None
        for raw_base in candidates:
            b = raw_base.expanduser().resolve()
            try:
                b.mkdir(parents=True, exist_ok=True)
            except Exception:
                continue
            probe = b / ".opencode_fsm_write_probe"
            try:
                probe.write_text("ok\n", encoding="utf-8")
                probe.unlink()
                base2 = b
                break
            except Exception:
                continue
        if base2 is None:
            base2 = (Path(tempfile.gettempdir()) / "opencode_fsm_targets").expanduser().resolve()
        base = base2
    base = base.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    if hf:
        namespace, name = hf
        if clones_dir is not None:
            prefix = f"hf_{namespace}_{name}"
            reused = _find_reusable_clone(base, prefix=prefix)
            if reused is not None:
                return PreparedRepo(repo=reused.resolve(), cloned_from=raw)

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        dest = base / f"hf_{namespace}_{name}_{ts}"
        env = dict(os.environ)
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        ok, detail = _download_hf_dataset_snapshot(namespace=namespace, name=name, dest=dest, env=env)
        if not ok:
            raise RuntimeError(f"hf dataset download failed: {detail}")
        return PreparedRepo(repo=dest.resolve(), cloned_from=raw)

    url = normalize_repo_url(raw)
    s2 = url.strip().rstrip("/")
    if s2.startswith("git@") and ":" in s2:
        s2 = s2.split(":", 1)[1]
    if "://" in s2:
        s2 = s2.split("://", 1)[1]
    s2 = s2.rstrip(".git")
    parts2 = [p for p in re.split(r"[/:]", s2) if p]
    name2 = parts2[-1] if parts2 else "repo"
    owner2 = parts2[-2] if len(parts2) >= 2 else "remote"
    slug = f"{owner2}_{name2}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
    slug = slug[:80]
    if clones_dir is not None:
        reused = _find_reusable_clone(base, prefix=slug)
        if reused is not None:
            return PreparedRepo(repo=reused.resolve(), cloned_from=url)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dest = base / f"{slug}_{ts}"
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    # Depth=1 keeps it fast; users can re-run with a local clone if needed.
    cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    try:
        clone_timeout = float(os.environ.get("OPENCODE_FSM_GIT_CLONE_TIMEOUT_SECONDS") or 90)
    except Exception:
        clone_timeout = 90.0
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=clone_timeout)
        rc = int(proc.returncode)
        out = proc.stdout or ""
        err = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        rc = 124
        out_raw = getattr(e, "stdout", "") or ""
        err_raw = getattr(e, "stderr", "") or ""
        if isinstance(out_raw, bytes):
            out = out_raw.decode("utf-8", errors="replace")
        else:
            out = str(out_raw)
        if isinstance(err_raw, bytes):
            err = err_raw.decode("utf-8", errors="replace")
        else:
            err = str(err_raw)
        err = (err + "\n" if err else "") + f"git_clone_timeout_exceeded: {clone_timeout}s"

    if rc != 0:
        # Clean up partial clones to avoid confusing fallbacks.
        shutil.rmtree(dest, ignore_errors=True)

        owner_repo = None
        s3 = str(url or "").strip()
        if s3:
            # https://github.com/<owner>/<repo>(.git)?
            m = re.match(r"^https?://([^/]+)/([^/]+)/([^/]+?)(?:\\.git)?/?$", s3)
            if m and str(m.group(1) or "").lower() in _GITHUB_ARCHIVE_HOSTS:
                owner_repo = (m.group(2), m.group(3))
            else:
                # git@github.com:<owner>/<repo>(.git)?
                m = re.match(r"^git@([^:]+):([^/]+)/([^/]+?)(?:\\.git)?$", s3)
                if m and str(m.group(1) or "").lower() in _GITHUB_ARCHIVE_HOSTS:
                    owner_repo = (m.group(2), m.group(3))
                else:
                    # ssh://git@github.com/<owner>/<repo>(.git)?
                    m = re.match(r"^ssh://git@([^/]+)/([^/]+)/([^/]+?)(?:\\.git)?/?$", s3)
                    if m and str(m.group(1) or "").lower() in _GITHUB_ARCHIVE_HOSTS:
                        owner_repo = (m.group(2), m.group(3))
        if owner_repo:
            owner, name = owner_repo
            ok, detail = _archive_clone_github(owner=owner, repo=name, dest=dest, env=env)
            if ok:
                return PreparedRepo(repo=dest.resolve(), cloned_from=url)

            raise RuntimeError(
                "git clone failed; GitHub archive fallback also failed\n"
                f"git_cmd: {' '.join(cmd)}\n"
                f"git_rc: {rc}\n"
                f"git_stdout: {out[-2000:]}\n"
                f"git_stderr: {err[-2000:]}\n"
                f"archive_detail: {detail}\n"
            )

        raise RuntimeError(
            "git clone failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"rc: {rc}\n"
            f"stdout: {out[-2000:]}\n"
            f"stderr: {err[-2000:]}\n"
        )

    # Make sure local commits (if any) won't fail due to missing identity.
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "opencode-fsm"], env=env, check=False)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "opencode-fsm@example.com"], env=env, check=False)

    return PreparedRepo(repo=dest.resolve(), cloned_from=url)
