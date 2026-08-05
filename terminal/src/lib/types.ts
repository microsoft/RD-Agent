export interface OHLCVBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SymbolInfo {
  symbol: string;
  baseCoin: string;
  quoteCoin: string;
  status: string;
}

export interface Ticker {
  symbol: string;
  lastPrice: number;
  price24hPcnt: number;
  volume24h: number;
  highPrice24h: number;
  lowPrice24h: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  brokers: string[];
  testnet: boolean;
}

export interface SymbolsResponse {
  broker: string;
  symbols: SymbolInfo[];
}

export interface KlinesResponse {
  broker: string;
  symbol: string;
  interval: string;
  bars: OHLCVBar[];
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export const INTERVAL_OPTIONS = [
  { label: "1m", value: "1" },
  { label: "5m", value: "5" },
  { label: "15m", value: "15" },
  { label: "1h", value: "60" },
  { label: "4h", value: "240" },
  { label: "1D", value: "D" },
] as const;
