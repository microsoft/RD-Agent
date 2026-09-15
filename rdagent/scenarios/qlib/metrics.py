"""Names of the Qlib metrics that RD-Agent reads back from a finished workflow.

The keys are produced by Qlib, not by RD-Agent: ``IC``/``ICIR``/``Rank IC``/``Rank ICIR`` come from
``SigAnaRecord`` and the ``1day.<group>.<metric>`` keys from ``PortAnaRecord``. Every place that indexes an
``experiment.result`` Series by metric name should import the constant instead of spelling the key out, so a
typo in one copy cannot silently disagree with another (see #1451).
"""

IC_KEY = "IC"
ICIR_KEY = "ICIR"
RANK_IC_KEY = "Rank IC"
RANK_ICIR_KEY = "Rank ICIR"

# Portfolio analysis of the excess return over the benchmark, after transaction cost.
ARR_KEY = "1day.excess_return_with_cost.annualized_return"
IR_KEY = "1day.excess_return_with_cost.information_ratio"
MDD_KEY = "1day.excess_return_with_cost.max_drawdown"  # Qlib reports drawdown as a number <= 0
