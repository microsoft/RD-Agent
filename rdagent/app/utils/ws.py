from typing import Optional

import typer

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.utils.agent.tpl import T

app = typer.Typer(help="Run data-science environment commands.")


@app.command()
def run(competition: str, cmd: str, local_path: str = "./"):
    """
    Launch the data-science environment for a specific competition and run the
    provided command.

    Example:
        1) start the container:
        dotenv run -- python -m rdagent.app.utils.ws nomad2018-predict-transparent-conductors "sleep 3600" --local-path your_workspace

        2) then run the following command to enter the latest container:
        - docker exec -it `docker ps --filter 'status=running' -l --format '{{.Names}}'` bash
        Or you can attach to the container by specifying the container name (find it in the run info)
        - docker exec -it sweet_robinson bash

    Arguments:
        competition: The competition slug/folder name.
        cmd: The shell command or script entry point to execute inside
             the environment.
    """
    data_path = DS_RD_SETTING.local_data_path

    data_path = (
        f"{data_path}/{competition}" if DS_RD_SETTING.sample_data_by_LLM else f"{data_path}/sample/{competition}"
    )
    target_path = T("scenarios.data_science.share:scen.input_path").r()
    extra_volumes = {data_path: target_path}

    # Don't set time limitation and always disable cache
    env = get_ds_env(
        extra_volumes=extra_volumes,
        running_timeout_period=None,
        enable_cache=False,
    )

    env.run(entry=cmd, local_path=local_path)


if __name__ == "__main__":  # pragma: no cover
    app()
