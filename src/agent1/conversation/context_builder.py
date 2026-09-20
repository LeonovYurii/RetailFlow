from agent1.conversation.message import (
    MessageRole,
)


class ContextBuilder:
    """
    Build textual context from conversation state.

    The complete conversation is stored in ConversationState.
    This class controls how much of that history is passed
    to downstream components.
    """

    def __init__(
        self,
        max_messages: int = 6,
        include_assistant: bool = True,
        include_tool_messages: bool = False,
    ):
        if max_messages < 1:
            raise ValueError(
                "max_messages must be at least 1."
            )

        self.max_messages = max_messages
        self.include_assistant = include_assistant
        self.include_tool_messages = (
            include_tool_messages
        )

    def _select_messages(
        self,
        conversation_state,
    ):
        messages = (
            conversation_state.get_recent_messages(
                self.max_messages
            )
        )

        selected = []

        for message in messages:
            if (
                message.role == MessageRole.TOOL
                and not self.include_tool_messages
            ):
                continue

            if (
                message.role == MessageRole.ASSISTANT
                and not self.include_assistant
            ):
                continue

            selected.append(message)

        return selected

    def build(
        self,
        conversation_state,
    ):
        """
        Build general conversational context.
        """

        messages = self._select_messages(
            conversation_state
        )

        parts = []

        for message in messages:
            parts.append(
                f"{message.role.value}: "
                f"{message.content}"
            )

        return "\n".join(parts)

    def build_for_classifier(
        self,
        conversation_state,
    ):
        """
        Build context for intent classification.

        For a single-turn request, this effectively returns
        the current user message.

        For a clarification follow-up, it keeps enough recent
        conversation to interpret short responses correctly.
        """

        messages = self._select_messages(
            conversation_state
        )

        if not messages:
            return ""

        # If there is only one user message,
        # do not add unnecessary role prefixes.
        if (
            len(messages) == 1
            and messages[0].role == MessageRole.USER
        ):
            return messages[0].content

        parts = []

        for message in messages:
            parts.append(
                f"{message.role.value}: "
                f"{message.content}"
            )

        return "\n".join(parts)

    def build_for_analyzer(
        self,
        conversation_state,
    ):
        """
        Build richer context for the request analyzer.
        """

        context = self.build(
            conversation_state
        )

        if (
            conversation_state.pending_clarification
            is None
        ):
            return context

        pending_section = (
            "\n\nPending clarification:\n"
            f"{conversation_state.pending_clarification}"
        )

        return context + pending_section