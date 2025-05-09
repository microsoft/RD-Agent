import os
from litellm import completion 
from pydantic import BaseModel, Field
import dotenv

dotenv.load_dotenv(override=True)
messages = [{"role": "user", "content": "List 5 important events in the XIX century"}]

class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]

class EventsList(BaseModel):
    events: list[CalendarEvent] = Field(min_items=1)

resp = completion(
    model='azure/o3-mini',
    messages=messages,
    # response_format=EventsList
)

print("Received={}".format(resp))

from litellm import get_supported_openai_params

params = get_supported_openai_params(model='azure/o3-mini')

assert "response_format" in params