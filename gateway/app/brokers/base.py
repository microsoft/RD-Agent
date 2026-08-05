from abc import ABC, abstractmethod

from app.models.market import OHLCVBar, Symbol, Ticker

BROKER_REGISTRY: dict[str, type["BrokerAdapter"]] = {}


def register_broker(broker_id: str):
    def decorator(cls: type["BrokerAdapter"]) -> type["BrokerAdapter"]:
        BROKER_REGISTRY[broker_id] = cls
        return cls

    return decorator


def get_broker(broker_id: str, **kwargs) -> "BrokerAdapter":
    if broker_id not in BROKER_REGISTRY:
        raise KeyError(f"Unknown broker: {broker_id}")
    return BROKER_REGISTRY[broker_id](**kwargs)


class BrokerAdapter(ABC):
    broker_id: str
    market_type: str = "crypto"

    @abstractmethod
    async def get_symbols(self, category: str = "linear") -> list[Symbol]:
        raise NotImplementedError

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        category: str = "linear",
    ) -> list[OHLCVBar]:
        raise NotImplementedError

    @abstractmethod
    async def get_ticker(self, symbol: str, category: str = "linear") -> Ticker:
        raise NotImplementedError
