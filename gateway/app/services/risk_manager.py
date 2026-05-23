"""Risk management service (Phase 3).

Planned controls:
- kill switch: halt all new orders immediately
- max notional per symbol
- daily loss limit with auto kill-switch
- manual approval gate before live order execution
"""


class RiskManager:
    """Stub for Phase 3 execution risk controls."""

    def check_order(self, *args, **kwargs) -> bool:
        """Validate order against risk limits. Not enforced in PR #1."""
        raise NotImplementedError("RiskManager is a Phase 3 stub")

    def is_kill_switch_active(self) -> bool:
        """Return True if kill switch is engaged."""
        return False

    def activate_kill_switch(self, reason: str) -> None:
        """Engage kill switch and block new orders."""

    def deactivate_kill_switch(self) -> None:
        """Disengage kill switch after manual review."""
