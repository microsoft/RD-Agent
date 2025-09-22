"""
Here are a list of MCP servers.

The MCP server is a individual RESTful API. So the only following things are included in the folder:
- Settings.
  - e.g., mcp/<mcp_name>.py:class Settings(BaseSettings);  then it is initialized as a global variable SETTINGS.
  - It only defines the format of the settings in Python Class (i.e., Pydantic BaseSettings).
- health_check:
  - e.g., mcp/<mcp_name>.py:def health_check() -> bool;
"""
