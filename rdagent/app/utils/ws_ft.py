from typing import Optional

import typer

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.utils.agent.tpl import T

app = typer.Typer(help="Run LLM fine-tuning environment commands.")


@app.command()
def run(
    dataset: str,
    model: str,
    cmd: str,
    local_path: str = "./",
    mount_path: str | None = None,
):
    """
    Launch the LLM fine-tuning environment for a specific dataset and model, then run the
    provided command.

    Example:
        1) start the container:
        dotenv run -- python -m rdagent.app.utils.ws_ft alpaca_gpt4_zh qwen2-7b "sleep 3600" --local-path your_workspace

        2) then run the following command to enter the latest container:
        - docker exec -it `docker ps --filter 'status=running' -l --format '{{.Names}}'` bash
        Or you can attach to the container by specifying the container name (find it in the run info)
        - docker exec -it sweet_robinson bash

    Arguments:
        dataset: The dataset name for fine-tuning.
        model: The base model name for fine-tuning.
        cmd: The shell command or script entry point to execute inside
             the environment.
    """
    # Build data and model paths based on FT_RD_SETTING configuration
    if FT_RD_SETTING.file_path:
        dataset_path = f"{FT_RD_SETTING.file_path}/dataset/{dataset}"
        model_path = f"{FT_RD_SETTING.file_path}/model/{model}"
        extra_volumes = {
            dataset_path: "/workspace/llm_finetune/data",
            model_path: "/workspace/llm_finetune/model",
        }
    else:
        # Fallback to current directory structure
        extra_volumes = {
            f"./dataset/{dataset}": "/workspace/llm_finetune/data",
            f"./model/{model}": "/workspace/llm_finetune/model",
        }

    # Don't set time limitation and always disable cache
    env = get_ft_env(
        extra_volumes=extra_volumes,
        running_timeout_period=None,
        enable_cache=False,
    )

    if mount_path is not None:
        env.conf.mount_path = mount_path

    env.run(entry=cmd, local_path=local_path)


if __name__ == "__main__":  # pragma: no cover
    app()
