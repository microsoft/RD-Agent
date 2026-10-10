from types import SimpleNamespace

from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace


def _trace(dag_parent: list[tuple[int, ...]]) -> DSTrace:
    trace = DSTrace(scen=None)
    trace.dag_parent = dag_parent
    trace.hist = [(f"exp{idx}", None) for idx in range(len(dag_parent))]
    return trace


def test_get_sibling_exps_when_a_node_has_two_children():
    # 0 -> 1 and 0 -> 2: the leaves are 1 and 2
    trace = _trace([(), (0,), (0,)])

    assert trace.get_sibling_exps(current_selection=trace.NEW_ROOT) == ["exp1", "exp2"]


def test_get_sibling_exps_when_uncommitted_experiments_share_a_parent():
    # 0 -> 1 committed, and two uncommitted experiments expanding node 1
    trace = _trace([(), (0,)])
    first, second = SimpleNamespace(local_selection=(1,)), SimpleNamespace(local_selection=(1,))
    trace.uncommitted_experiments = {2: first, 3: second}

    assert trace.get_sibling_exps(current_selection=trace.NEW_ROOT) == [first, second]


def test_get_sibling_exps_ignores_the_current_selection():
    trace = _trace([(), (0,), (0,)])

    assert trace.get_sibling_exps(current_selection=(1,)) == ["exp2"]
