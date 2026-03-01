from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _file = os.path.abspath(__file__)
    _SCRIPT_DIR = os.path.dirname(_file)
    _ROOT = os.path.dirname(_SCRIPT_DIR)
    root_s = str(_ROOT)
    script_s = str(_SCRIPT_DIR)
    try:
        while script_s in sys.path:
            sys.path.remove(script_s)
        while root_s in sys.path:
            sys.path.remove(root_s)
    except Exception:
        pass
    sys.path.insert(0, root_s)

import json
import random
import re
import textwrap
import time
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    from runner._util import (  # type: ignore
        _ensure_openai_v1_base,
        _find_hf_test_parquet,
        _parse_json_str_list,
        _read_json_object,
    )
else:
    from ._util import (
        _ensure_openai_v1_base,
        _find_hf_test_parquet,
        _parse_json_str_list,
        _read_json_object,
    )


def _chat_completion(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    timeout_seconds: int,
    max_tokens: int | None = None,
) -> str:
    url = _ensure_openai_v1_base(base_url) + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if max_tokens is None:
        raw = (os.environ.get("OPENCODE_FSM_MAX_TOKENS") or "").strip()
        n = None
        if raw:
            try:
                n = int(raw)
            except Exception:
                n = None
        max_tokens = n if isinstance(n, int) and n > 0 else 256
    max_tokens = max(1, int(max_tokens))
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read()
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    return content if isinstance(content, str) else ""


_RE_NUM = re.compile(r"-?\\d+(?:,\\d{3})*(?:\\.\\d+)?(?:/\\d+(?:\\.\\d+)?)?")

_RE_FINAL_LINE = re.compile(r"(?im)^\\s*final\\s*[:：]\\s*(?P<ans>.+?)\\s*$")


def _can_use_remote_completion(base_url: str | None, api_key: str | None) -> bool:
    if not base_url:
        return False
    if api_key:
        return True
    lowered = base_url.strip().lower()
    trusted_prefixes = (
        "http://127.",
        "https://127.",
        "http://localhost",
        "https://localhost",
        "http+unix://",
        "unix://",
    )
    return lowered.startswith(trusted_prefixes)


def _maybe_rollout_hf_qa_parquet(
    repo_root: Path,
    *,
    artifacts_dir: Path,
    base_url: str,
    api_key: str | None,
    model: str,
    mode: str,
    limit: int,
) -> tuple[bool, dict[str, Any]]:
    """Best-effort: if repo_root looks like an HF dataset snapshot, generate QA rollout samples.

    This intentionally avoids dataset-id hardcoding. It only triggers when:
    - `data/hf_manifest.json` exists (repo_resolver HF snapshot marker)
    - a test parquet exists
    - the parquet has columns: `question`, `answer`
    """
    manifest_path = (repo_root / "data" / "hf_manifest.json").resolve()
    if not manifest_path.exists() or _read_json_object(manifest_path) is None:
        return False, {}

    parquet_path = _find_hf_test_parquet(repo_root)
    if parquet_path is None:
        return False, {"reason": "hf_manifest_present_but_no_test_parquet"}

    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        return False, {"reason": f"pyarrow_unavailable: {e}"}

    try:
        table = pq.read_table(parquet_path, columns=["question", "answer"])
    except Exception as e:
        return False, {"reason": f"failed_to_read_parquet: {e}"}

    try:
        if (
            int(limit) > 0
            and getattr(table, "num_rows", 0)
            and int(table.num_rows) > int(limit)
        ):
            table = table.slice(0, int(limit))
    except Exception:
        pass

    try:
        rows = table.to_pylist()
    except Exception as e:
        return False, {"reason": f"failed_to_convert_parquet: {e}"}

    if not rows:
        return False, {"reason": "no_rows_in_test_parquet"}

    template = (
        "Answer the following question. You may include reasoning, but put the final answer on the last line as:\n"
        "FINAL: <answer>\n\n"
        "{prompt}"
    )

    samples_path = (
        artifacts_dir / f"rollout_samples_{int(time.time())}.jsonl"
    ).resolve()
    correct = 0
    lines = 0
    errors: list[str] = []

    with samples_path.open("w", encoding="utf-8") as out:
        for r in rows:
            if not isinstance(r, dict):
                continue
            q = str(r.get("question") or "").strip()
            a = str(r.get("answer") or "").strip()
            if not q or not a:
                continue

            prompt = template.format(prompt=q)
            try:
                completion = _chat_completion(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    timeout_seconds=120,
                )
            except Exception as e:
                errors.append(str(e))
                completion = ""

            # Extract best-effort final answer line (FINAL: ...). Fall back to the last non-empty line.
            t = str(completion or "").strip()
            if not t:
                pred = ""
            else:
                m = _RE_FINAL_LINE.search(t)
                if m:
                    ans = str(m.group("ans") or "").strip()
                    pred = ans if ans else ""
                else:
                    lines2 = [ln.strip() for ln in t.splitlines() if ln.strip()]
                    if not lines2:
                        pred = t
                    else:
                        last = lines2[-1]
                        pred = re.sub(r"(?i)^final\\s*[:：]\\s*", "", last).strip()

            t = str(a or "").strip()
            if not t:
                gold = ""
            else:
                m = _RE_FINAL_LINE.search(t)
                if m:
                    ans = str(m.group("ans") or "").strip()
                    gold = ans if ans else ""
                else:
                    lines2 = [ln.strip() for ln in t.splitlines() if ln.strip()]
                    if not lines2:
                        gold = t
                    else:
                        last = lines2[-1]
                        gold = re.sub(r"(?i)^final\\s*[:：]\\s*", "", last).strip()

            nums = _RE_NUM.findall(str(pred or ""))
            pred_num = str(nums[-1] or "").strip() if nums else ""
            nums = _RE_NUM.findall(str(gold or ""))
            gold_num = str(nums[-1] or "").strip() if nums else ""
            if pred_num and gold_num:
                pred_ss = str(pred_num or "").strip().replace(",", "")
                gold_ss = str(gold_num or "").strip().replace(",", "")
                try:
                    fp = Fraction(pred_ss)
                except Exception:
                    fp = None
                try:
                    fg = Fraction(gold_ss)
                except Exception:
                    fg = None
                if fp is not None and fg is not None:
                    is_ok = fp == fg
                else:
                    is_ok = pred_ss == gold_ss
            else:
                pred_norm = re.sub(r"\\s+", " ", str(pred or "").strip().lower())
                gold_norm = re.sub(r"\\s+", " ", str(gold or "").strip().lower())
                is_ok = pred_norm == gold_norm

            reward = 1.0 if is_ok else 0.0
            correct += int(is_ok)
            lines += 1
            out.write(
                json.dumps(
                    {"prompt": prompt, "completion": completion, "reward": reward},
                    ensure_ascii=False,
                )
                + "\n"
            )

    if lines <= 0:
        return False, {"reason": "no_usable_rows_for_rollout"}

    rollout = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "ok": True,
        "mode": mode,
        "counts": {"samples": lines, "correct": correct, "errors": len(errors)},
        "dataset": {"parquet": str(parquet_path)},
        "paths": {"samples_jsonl": str(samples_path)},
    }
    if errors:
        rollout["errors"] = errors[:3]
    return True, rollout


def _read_text_excerpt(path: Path, *, limit: int = 1200) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    snippet = raw.strip()
    if not snippet:
        return ""
    snippet = re.sub(r"\s+", " ", snippet)
    if len(snippet) > int(limit):
        snippet = snippet[: int(limit)].rstrip() + "..."
    return snippet


def _build_repo_summary(repo_root: Path) -> str:
    snippets: list[str] = []
    for rel in ("description.md", "README.md", "readme.md"):
        p = (repo_root / rel).resolve()
        if p.exists():
            excerpt = _read_text_excerpt(p)
            if excerpt:
                snippets.append(f"{rel} excerpt: {excerpt}")
    train_path = (repo_root / "data" / "train.jsonl").resolve()
    if train_path.exists():
        try:
            with train_path.open("r", encoding="utf-8", errors="replace") as f:
                first_line = ""
                for line in f:
                    candidate = line.strip()
                    if candidate:
                        first_line = candidate
                        break
        except Exception:
            first_line = ""
        if first_line:
            try:
                obj = json.loads(first_line)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                question = str(obj.get("question") or "").strip()
                task_id = str(obj.get("task_id") or "").strip()
                if question:
                    question_excerpt = textwrap.shorten(
                        re.sub(r"\s+", " ", question), width=280, placeholder="..."
                    )
                    if task_id:
                        snippets.append(f"First task ({task_id}): {question_excerpt}")
                    else:
                        snippets.append(f"First task prompt: {question_excerpt}")
    if not snippets:
        return "Repository context unavailable; inspect workspace files under data/ and output/."
    return "\n".join(snippets)


def _fallback_completion(prompt: str, repo_summary: str) -> str:
    prompt_clean = re.sub(r"\s+", " ", str(prompt or "").strip())
    summary = repo_summary.strip() or "Repository context unavailable."
    return (
        "Automated fallback response (no inference endpoint detected).\n"
        f"Prompt summary: {prompt_clean[:400]}\n\n"
        f"Repository context:\n{summary}\n\n"
        "Getting started: review description.md for benchmark details, inspect data/train.jsonl for HumanEval tasks, "
        "and run .opencode_fsm/stages/evaluation.sh to execute hinted evaluations or reward averaging."
    )


def main() -> int:
    repo_root = Path(os.environ.get("OPENCODE_FSM_REPO_ROOT") or ".").resolve()
    artifacts_dir = Path(
        os.environ.get("OPENCODE_FSM_ARTIFACTS_DIR")
        or (repo_root / ".opencode_fsm" / "artifacts")
    )
    if not artifacts_dir.is_absolute():
        artifacts_dir = (repo_root / artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mode = (os.environ.get("OPENCODE_EVAL_MODE") or "smoke").strip().lower() or "smoke"
    try:
        limit = int(
            os.environ.get("OPENCODE_EVAL_LIMIT") or (64 if mode == "full" else 8)
        )
    except Exception:
        limit = 8
    limit = max(1, int(limit))

    base_url = (
        os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or ""
    ).strip()
    if base_url:
        base_url = base_url.rstrip("/")
    else:
        runtime_path = (os.environ.get("OPENCODE_RUNTIME_ENV_PATH") or "").strip()
        if runtime_path:
            p = Path(runtime_path)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                obj = None
            if isinstance(obj, dict):
                inf = obj.get("inference")
                if isinstance(inf, dict):
                    b = str(inf.get("openai_base_url") or "").strip()
                    if b:
                        base_url = b.rstrip("/")
                if not base_url:
                    svc = obj.get("service")
                    if isinstance(svc, dict):
                        b2 = str(svc.get("base_url") or "").strip()
                        if b2:
                            base_url = b2.rstrip("/")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None
    if not base_url and api_key:
        base_url = "https://api.openai.com/v1"
    can_use_remote = _can_use_remote_completion(base_url, api_key)
    if not can_use_remote:
        base_url = ""
    model = (os.environ.get("OPENCODE_LLM_MODEL") or "").strip()
    if not model:
        raise SystemExit("missing_model: set OPENCODE_LLM_MODEL")

    # HF dataset snapshot support: if detected, generate QA samples with rewards.
    ok_ds, ds_rollout = _maybe_rollout_hf_qa_parquet(
        repo_root,
        artifacts_dir=artifacts_dir,
        base_url=base_url,
        api_key=api_key,
        model=model,
        mode=mode,
        limit=limit,
    )
    if ok_ds:
        rollout_path = (repo_root / ".opencode_fsm" / "rollout.json").resolve()
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_path.write_text(
            json.dumps(ds_rollout, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    # Default: generic, bounded "repo understanding" prompts (cap for safety).
    prompt_limit = max(1, min(int(limit), 32))
    if mode == "smoke":
        prompt_limit = min(prompt_limit, 1)
    hints = _parse_json_str_list(os.environ.get("OPENCODE_FSM_HINTS_JSON"))
    prompts: list[str] = []
    for h in hints[: max(0, prompt_limit)]:
        prompts.append(f"Explain how to run this command and what it does:\n\n{h}")

    if len(prompts) < prompt_limit:
        readme = ""
        for cand in ("README.md", "readme.md", "README.txt"):
            p = (repo_root / cand).resolve()
            if p.exists():
                try:
                    readme = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    readme = ""
                if len(readme) > 6000:
                    readme = readme[:6000]
                break
        if readme:
            chunks = [c.strip() for c in readme.split("\n\n") if c.strip()]
            random.shuffle(chunks)
            for c in chunks:
                if len(prompts) >= prompt_limit:
                    break
                excerpt = c[:800]
                prompts.append(f"Answer based on this repo excerpt:\n\n{excerpt}")

    if not prompts:
        repo_name = repo_root.name
        prompts = [
            f"Describe the purpose of the repository `{repo_name}` and how to get started."
        ]
    prompts = prompts[:prompt_limit]
    samples_path = (
        artifacts_dir / f"rollout_samples_{int(time.time())}.jsonl"
    ).resolve()

    samples_written = 0
    errors_count = 0
    with samples_path.open("w", encoding="utf-8") as f:
        repo_summary = _build_repo_summary(repo_root)
        for prompt in prompts:
            completion = ""
            reward = 0.0
            if can_use_remote:
                try:
                    completion = _chat_completion(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        prompt=prompt,
                        timeout_seconds=30,
                    )
                    reward = 1.0 if str(completion).strip() else 0.0
                except (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    TimeoutError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    completion = ""
                    errors_count += 1
            if not str(completion or "").strip():
                completion = _fallback_completion(prompt, repo_summary)
                reward = 0.0
            obj = {"prompt": prompt, "completion": completion, "reward": reward}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            samples_written += 1

    rollout_path = (repo_root / ".opencode_fsm" / "rollout.json").resolve()
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "ok": True,
        "mode": mode,
        "counts": {"samples": samples_written, "errors": int(errors_count)},
        "paths": {"samples_jsonl": str(samples_path)},
    }
    rollout_path.write_text(
        json.dumps(rollout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
