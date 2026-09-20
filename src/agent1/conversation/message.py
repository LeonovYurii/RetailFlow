from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageType(str, Enum):
    MESSAGE = "message"
    CLARIFICATION = "clarification"
    TOOL_RESULT = "tool_result"


@dataclass
class ChatMessage:
    role: MessageRole
    content: str

    message_type: MessageType = MessageType.MESSAGE
    metadata: dict[str, Any] = field(default_factory=dict)