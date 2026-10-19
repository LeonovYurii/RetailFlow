from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standard result returned by an Agent1 tool."""

    success: bool
    action: str
    data: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None