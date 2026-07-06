"""
rdagent-run-qlib — run a qlib backtest as a deterministic, LLM-free tool.

This is the "run" hand of the agent-driven loop (ADR 0002). The agent proposes a
hypothesis and writes factor/model code; this tool executes it against qlib inside
the vendored Docker image and returns the metric Series (`exp.result`). No LLM, no
.env, no ANTHROPIC_API_KEY.

It wraps `QlibFBWorkspace.execute()` (which is already LLM-free) and bakes in the
two environment fixes needed on this machine, discovered during slice-1 bring-up:
  1. Force env_type=docker (the default is conda, which would auto-build a heavy
     `rdagent4qlib` env; the `local_qlib:latest` image is already provisioned).
  2. Pass MLFLOW_ALLOW_FILE_STORE=true into the container (its newer mlflow refuses
     the file-store backend otherwise).

Examples
--------
Baseline (Alpha20 features, LGBM), short segment:
    python -m agent_loop.run_qlib --conf conf_baseline.yaml \
        --train_start 2015-01-01 --train_end 2016-12-31 \
        --valid_start 2017-01-01 --valid_end 2017-12-31 \
        --test_start 2018-01-01 --test_end 2018-12-31 --out /tmp/res.json

Custom factor(s): drop a `combined_factors_df.parquet` (MultiIndex [datetime,
instrument] rows; columns under a top-level "feature" level) and run:
    python -m agent_loop.run_qlib --conf conf_combined_factors.yaml \
        --factors /path/to/combined_factors_df.parquet --out /tmp/res.json
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import fire


def _rdagent_root() -> Path:
    # rdagent is a namespace package (__file__ is None); resolve via __path__.
    import rdagent

    return Path(list(rdagent.__path__)[0])


def _template_dir(template: str) -> Path:
    root = _rdagent_root() / "scenarios" / "qlib" / "experiment"
    mapping = {
        "factor": root / "factor_template",
        "model": root / "model_template",
    }
    if template not in mapping:
        raise ValueError(f"template must be one of {list(mapping)}, got {template!r}")
    d = mapping[template]
    if not d.is_dir():
        raise FileNotFoundError(f"template folder not found: {d}")
    return d


def _load_features(features: str | None) -> dict[str, str]:
    if features is None:
        from rdagent.utils.qlib import ALPHA20

        return dict(ALPHA20)
    p = Path(features)
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{p} must be a JSON object of name -> expression")
    return data


def run(
    conf: str = "conf_baseline.yaml",
    template: str = "factor",
    factors: str | None = None,
    features: str | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    valid_start: str | None = None,
    valid_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    env: str = "docker",
    gpu: bool = False,
    timeout: int = 3600,
    out: str | None = None,
) -> dict[str, Any]:
    """Run a qlib backtest and return {metric_name: value}.

    Parameters mirror the qlib template segments. Unset date args fall back to the
    template defaults (train 2008-2014, valid 2015-2016, test 2017+).
    """
    # --- force the provisioned execution path (must be set before QTDockerEnv build)
    os.environ["QLIB_DOCKER_ENABLE_GPU"] = str(bool(gpu))
    os.environ["QLIB_DOCKER_RUNNING_TIMEOUT_PERIOD"] = str(int(timeout))

    from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
    from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

    MODEL_COSTEER_SETTINGS.env_type = env  # execute() reads this at call time

    template_dir = _template_dir(template)
    conf_path = template_dir / conf
    if not conf_path.exists():
        available = sorted(p.name for p in template_dir.glob("conf*.yaml"))
        raise FileNotFoundError(f"{conf!r} not in {template_dir}. Available: {available}")

    feats = _load_features(features)
    run_env = {
        "PYTHONPATH": "./",
        # Container ships a newer mlflow that refuses the file store unless opted in.
        "MLFLOW_ALLOW_FILE_STORE": "true",
        "feature_names": str(list(feats.keys())),
        "feature_expressions": str(list(feats.values())),
    }
    for k, v in {
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
    }.items():
        if v is not None:
            run_env[k] = v

    ws = QlibFBWorkspace(template_folder_path=template_dir)
    if factors is not None:
        src = Path(factors)
        if not src.exists():
            raise FileNotFoundError(f"--factors parquet not found: {src}")
        shutil.copy(src, ws.workspace_path / "combined_factors_df.parquet")

    print(f"[run-qlib] env={env} gpu={gpu} conf={conf} workspace={ws.workspace_path}")
    result, stdout = ws.execute(qlib_config_name=conf, run_env=run_env)

    if result is None:
        tail = (stdout or "")[-2000:]
        raise RuntimeError(f"qlib run produced no result. stdout tail:\n{tail}")

    metrics = {str(k): float(v) for k, v in result.items()}
    print("\n==== metrics ====")
    for k in sorted(metrics):
        print(f"{k:52s} {metrics[k]:+.6f}")

    payload = {
        "workspace": str(ws.workspace_path),
        "conf": conf,
        "segments": {
            k: run_env.get(k)
            for k in ("train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end")
        },
        "metrics": metrics,
    }
    if out is not None:
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\n[run-qlib] wrote {out}")
    return payload


if __name__ == "__main__":
    fire.Fire(run)
