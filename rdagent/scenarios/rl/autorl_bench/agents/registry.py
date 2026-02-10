"""
Agent Registry
"""
import yaml
from dataclasses import dataclass
from pathlib import Path

AGENTS_DIR = Path(__file__).parent


@dataclass
class Agent:
    id: str
    name: str
    start: Path
    env_vars: dict = None
    
    def __post_init__(self):
        self.env_vars = self.env_vars or {}


def get_agent(agent_id: str) -> Agent:
    agent_dir = AGENTS_DIR / agent_id
    config_file = agent_dir / "config.yaml"
    
    if not config_file.exists():
        raise ValueError(f"Agent not found: {agent_id}")
    
    data = yaml.safe_load(config_file.read_text())
    
    return Agent(
        id=agent_id,
        name=data.get("name", agent_id),
        start=agent_dir / data.get("start", "start.sh"),
        env_vars=data.get("env_vars", {}),
    )


def list_agents() -> list[str]:
    return [d.name for d in AGENTS_DIR.iterdir() if d.is_dir() and (d / "config.yaml").exists()]
