"""Broker-related exceptions mapped to HTTP status codes in routers."""


class BrokerError(Exception):
    """Base broker error."""


class BrokerNotFoundError(BrokerError):
    """Unknown symbol or resource."""


class BrokerUpstreamError(BrokerError):
    """Upstream exchange API failure."""


class BrokerRateLimitError(BrokerError):
    """Rate limit exceeded."""
