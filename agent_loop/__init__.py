"""
agent_loop — deterministic, LLM-free tooling for the agent-driven quant R&D loop.

This package is intentionally kept OUTSIDE the vendored `rdagent/` tree so that
periodic upstream merges of microsoft/RD-Agent stay cheap (see ADR 0002). It wraps
RD-Agent's execution machinery (Dockerized qlib backtests, the trace ledger, and a
statistical guardrail) as tools that *this session's agent* drives directly — the
agent is the brain (propose / code / judge); these tools are the hands.

Nothing in here calls an LLM. No ANTHROPIC_API_KEY / LiteLLM is required.
"""
