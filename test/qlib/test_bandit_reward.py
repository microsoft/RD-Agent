from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from rdagent.scenarios.qlib.proposal.bandit import EnvController, Metrics


@pytest.mark.offline
def test_record_penalizes_drawdown_without_flipping_context():
    controller = EnvController()
    metrics = Metrics(ic=0.05, icir=0.4, rank_ic=0.06, rank_icir=0.5, arr=0.12, ir=1.1, sharpe=0.8)

    # Hold all other components, including the return/drawdown ratio, fixed
    # to isolate the drawdown penalty against a nonzero background reward.
    with patch.object(controller.bandit, "update", wraps=controller.bandit.update) as update:
        for mdd in (0.0, -0.1, -0.3):
            controller.record(replace(metrics, mdd=mdd), "factor")

    assert update.call_count == 3
    rewards = [call.args[2] for call in update.call_args_list]
    assert np.diff(rewards) == pytest.approx([-0.01, -0.02])
    assert controller.reward(Metrics()) == 0.0
    # The learner still receives positive drawdown magnitudes as context.
    assert [call.args[1][6] for call in update.call_args_list] == pytest.approx([0.0, 0.1, 0.3])
