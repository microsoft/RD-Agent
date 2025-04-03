from rdagent.core.proposal import Trace


def check_point_selector(trace: Trace) -> tuple[int] | None:
    return (-1, )


# TODO: more advanced selector
# TODO/Discussion: load selector function here or define selector class in `proposal.py`?
