from agent1.conversation.message import (
    ChatMessage,
    MessageRole,
    MessageType,
)


class ConversationState:
    """
    Store conversation history and temporary agent state
    for one conversation session.
    """

    def __init__(self):
        self.messages: list[ChatMessage] = []

        # Routing decision that caused the agent
        # to ask for clarification.
        self.pending_routing_decision = None

        # Optional clarification question currently
        # waiting for a user response.
        self.pending_clarification: str | None = None

    # ---------------------------------------------------------
    # Messages
    # ---------------------------------------------------------

    def add_message(
        self,
        role: MessageRole,
        content: str,
        message_type: MessageType = MessageType.MESSAGE,
        metadata: dict | None = None,
    ):
        message = ChatMessage(
            role=role,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )

        self.messages.append(message)

        return message

    def add_user_message(
        self,
        content: str,
        metadata: dict | None = None,
    ):
        return self.add_message(
            role=MessageRole.USER,
            content=content,
            metadata=metadata,
        )

    def add_assistant_message(
        self,
        content: str,
        metadata: dict | None = None,
    ):
        return self.add_message(
            role=MessageRole.ASSISTANT,
            content=content,
            metadata=metadata,
        )

    def add_clarification(
        self,
        content: str,
        routing_decision=None,
    ):
        self.pending_clarification = content
        self.pending_routing_decision = routing_decision

        return self.add_message(
            role=MessageRole.ASSISTANT,
            content=content,
            message_type=MessageType.CLARIFICATION,
        )

    def add_tool_result(
        self,
        content: str,
        metadata: dict | None = None,
    ):
        return self.add_message(
            role=MessageRole.TOOL,
            content=content,
            message_type=MessageType.TOOL_RESULT,
            metadata=metadata,
        )

    # ---------------------------------------------------------
    # State access
    # ---------------------------------------------------------

    def get_messages(self):
        return list(self.messages)

    def get_recent_messages(
        self,
        limit: int = 5,
    ):
        if limit < 1:
            return []

        return self.messages[-limit:]

    def has_pending_clarification(self):
        return (
            self.pending_routing_decision is not None
            or self.pending_clarification is not None
        )

    def resolve_clarification(self):
        """
        Clear temporary clarification state after
        the user's follow-up has been processed.
        """

        self.pending_routing_decision = None
        self.pending_clarification = None

    def clear(self):
        self.messages.clear()

        self.pending_routing_decision = None
        self.pending_clarification = None

    def __len__(self):
        return len(self.messages)