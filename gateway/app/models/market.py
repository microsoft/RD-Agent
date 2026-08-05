from pydantic import BaseModel, Field


class Symbol(BaseModel):
    symbol: str
    baseCoin: str
    quoteCoin: str
    status: str


class OHLCVBar(BaseModel):
    time: int = Field(..., description="Unix timestamp in seconds")
    open: float
    high: float
    low: float
    close: float
    volume: float


class Ticker(BaseModel):
    symbol: str
    lastPrice: float
    price24hPcnt: float
    volume24h: float
    highPrice24h: float
    lowPrice24h: float


class HealthResponse(BaseModel):
    status: str
    version: str
    brokers: list[str]
    testnet: bool


class SymbolsResponse(BaseModel):
    broker: str
    symbols: list[Symbol]


class KlinesResponse(BaseModel):
    broker: str
    symbol: str
    interval: str
    bars: list[OHLCVBar]
