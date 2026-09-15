import unittest
from unittest.mock import MagicMock

import pytest

from rdagent.components.workflow.rd_loop import RDLoop


class FakeExperiment:
    """Stand-in for an Experiment subclass: no dict-like .get() method."""


@pytest.mark.offline
class TestRDLoopRecord(unittest.TestCase):
    def test_record_uses_direct_exp_gen_when_returned_directly(self):
        # Some RDLoop subclasses (e.g. FactorReportLoop) override direct_exp_gen to
        # return the Experiment directly instead of the base class's {"exp_gen": Experiment}.
        # record() must fall back to that Experiment instead of calling .get() on it.
        fake_self = MagicMock()
        fake_self.LOOP_IDX_KEY = RDLoop.LOOP_IDX_KEY

        exp = FakeExperiment()
        feedback = object()
        prev_out = {
            "feedback": feedback,
            "direct_exp_gen": exp,
            RDLoop.LOOP_IDX_KEY: 0,
        }

        RDLoop.record(fake_self, prev_out)

        fake_self.trace.sync_dag_parent_and_hist.assert_called_once_with((exp, feedback), 0)

    def test_record_unwraps_direct_exp_gen_dict(self):
        # The base RDLoop.direct_exp_gen returns {"propose": ..., "exp_gen": Experiment}.
        fake_self = MagicMock()
        fake_self.LOOP_IDX_KEY = RDLoop.LOOP_IDX_KEY

        exp = FakeExperiment()
        feedback = object()
        prev_out = {
            "feedback": feedback,
            "direct_exp_gen": {"propose": "hypothesis", "exp_gen": exp},
            RDLoop.LOOP_IDX_KEY: 0,
        }

        RDLoop.record(fake_self, prev_out)

        fake_self.trace.sync_dag_parent_and_hist.assert_called_once_with((exp, feedback), 0)


if __name__ == "__main__":
    unittest.main()
