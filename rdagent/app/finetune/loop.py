import asyncio
from pathlib import Path
from typing import Optional

import fire
import typer
from typing_extensions import Annotated

from rdagent.app.finetune.conf import FINETUNE_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop


class FinetuneRDLoop(RDLoop):
    """R&D loop skeleton for the Finetune scenario.

    The actual logic depends on proper component implementation specified in
    ``FINETUNE_PROP_SETTING``. This class simply inherits :class:`RDLoop` and
    relies on that configuration.
    """

    # No additional methods for now – use RDLoop defaults
    pass


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    checkout_path: Optional[str] = None,
):
    """Entry point to launch the Finetune R&D loop.

    Parameters
    ----------
    path : str | None
        Resume from a previous session given its path.
    step_n / loop_n : int | None
        Limit the number of steps or loops to run, respectively.
    checkout : bool
        Whether to perform checkout (same semantics as other loops).
    checkout_path : str | None
        Alternative path to save new session outputs.
    """

    if checkout_path is not None:
        checkout = Path(checkout_path)

    # Start new loop or resume
    if path is None:
        loop = FinetuneRDLoop(FINETUNE_PROP_SETTING)
    else:
        loop = FinetuneRDLoop.load(path, checkout=checkout)

    # Run asynchronously like other loops
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n))


if __name__ == "__main__":
    fire.Fire(main)
