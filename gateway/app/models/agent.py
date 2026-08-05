from pydantic import BaseModel, Field


class ScenarioInfo(BaseModel):
    name: str
    target: str
    upload: bool
    developer: bool


class AgentRunResponse(BaseModel):
    id: str


class AgentControlRequest(BaseModel):
    id: str
    action: str = "stop"


class UserInteractionRequest(BaseModel):
    id: str
    payload: dict


class TraceMessage(BaseModel):
    tag: str
    timestamp: str | None = None
    content: dict | list | str | None = None
    loop_id: int | None = Field(default=None, alias="loop_id")

    model_config = {"populate_by_name": True, "extra": "allow"}
