from agent1.conversation.message import (
    ChatMessage,
    MessageRole,
    MessageType,
)

from typing import Any

from agent1.analysis.action_plan import ActionPlan, RequestType

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

        # A resolved action waiting for execution arguments.
        self.pending_action_plan: ActionPlan | None = None
        self.pending_arguments: dict[str, Any] = {}
        self.required_arguments: list[str] = []

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
        self.resolve_clarification()
        self.clear_pending_action()

    def __len__(self):
        return len(self.messages)

    def start_pending_action(
        self,
        action_plan: ActionPlan,
        required_arguments: list[str],
        arguments: dict[str, Any] | None = None,
    ):
        """Remember a resolved action and its execution requirements."""

        if action_plan.request_type != RequestType.SINGLE_ACTION:
            raise ValueError("Only a single action can be pending execution.")

        # The intended action is now resolved.
        self.resolve_clarification()

        self.pending_action_plan = action_plan.model_copy(deep=True)
        self.pending_arguments = dict(arguments or {})
        self.required_arguments = list(required_arguments)


    def update_pending_arguments(self, **arguments):
        """Store arguments collected for the pending action."""

        if self.pending_action_plan is None:
            raise ValueError("There is no pending action.")

        self.pending_arguments.update(arguments)


    def get_missing_arguments(self) -> list[str]:
        """Find required arguments that are absent or blank."""

        missing = []

        for name in self.required_arguments:
            value = self.pending_arguments.get(name)

            if value is None or (
                isinstance(value, str) and not value.strip()
            ):
                missing.append(name)

        return missing


    def has_pending_action(self) -> bool:
        return self.pending_action_plan is not None


    def clear_pending_action(self):
        """Clear execution state while preserving conversation history."""

        self.pending_action_plan = None
        self.pending_arguments.clear()
        self.required_arguments.clear()