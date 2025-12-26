import asyncio
from pathlib import Path
from typing import Optional

import fire
import typer
from typing_extensions import Annotated


from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger

from rdagent.app.agentic_sys.conf import ASYS_RD_SETTING
from rdagent.scenarios.agentic_sys.loop import AgenticSysRDLoop


def main(
    path: Optional[str] = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    checkout_path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
    competition="deepresearch",
    replace_timer=True,
    exp_gen_cls: Optional[str] = None,
):
    if not checkout_path is None:
        checkout = Path(checkout_path)

    if competition is not None:
        ASYS_RD_SETTING.competition = competition

    if not ASYS_RD_SETTING.competition:
        logger.error("Please specify competition name.")

    if path is None:
        agentic_sys_loop = AgenticSysRDLoop(ASYS_RD_SETTING)
    else:
        agentic_sys_loop: AgenticSysRDLoop = AgenticSysRDLoop.load(path, checkout=checkout, replace_timer=replace_timer)

    # replace exp_gen if we have new class
    if exp_gen_cls is not None:
        agentic_sys_loop.exp_gen = import_class(exp_gen_cls)(agentic_sys_loop.exp_gen.scen)

    asyncio.run(agentic_sys_loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))


if __name__ == "__main__":
    fire.Fire(main)
