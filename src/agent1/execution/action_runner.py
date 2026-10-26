from agent1.analysis.action_plan import ActionPlan
from agent1.conversation.state import ConversationState
from agent1.execution.action_executor import ActionExecutor
from agent1.execution.argument_collector import ArgumentCollector


class ActionRunner:
    """Manage argument collection and execution for supported actions."""

    def __init__(self, state: ConversationState, response_service=None):
        self.state = state
        self.collector = ArgumentCollector()
        self.executor = ActionExecutor()
        self.response_service = response_service
        self.last_tool_result = None
        self._request_text = ""

    def start(
        self,
        action_plan: ActionPlan,
        user_message: str,
    ) -> str:
        """Start an action already resolved by the analyzer."""

        if self.state.has_pending_action():
            raise ValueError(
                "Resolve or clear the pending action before starting another."
            )

        self.last_tool_result = None
        self._request_text = user_message
        required = self.collector.required_arguments(action_plan)
        arguments = self.collector.extract(user_message, action_plan)

        self.state.start_pending_action(
            action_plan=action_plan,
            required_arguments=required,
            arguments=arguments,
        )

        return self._advance()

    def resume(self, user_message: str) -> str:
        """Process a reply assumed to continue the pending action."""

        plan = self.state.pending_action_plan

        if plan is None:
            raise ValueError("There is no pending action to resume.")

        self._request_text += "\nFollow-up: " + user_message
        arguments = self.collector.extract(user_message, plan)
        self.state.update_pending_arguments(**arguments)

        return self._advance()

    def _advance(self) -> str:
        """Ask for missing arguments or execute the pending action."""

        missing = self.state.get_missing_arguments()

        if missing:
            if missing == ["order_id"]:
                reply = (
                    "Please provide one order number, such as ORD-123."
                )
            else:
                reply = f"Please provide: {', '.join(missing)}."

            return self._record_reply(reply)

        plan = self.state.pending_action_plan

        if plan is None:
            raise ValueError("There is no pending action to execute.")

        result = self.executor.execute(
            plan,
            **self.state.pending_arguments,
        )

        self.state.add_tool_result(
            content=result.model_dump_json(),
        )

        self.last_tool_result = result
        # Clear before generating: a response failure must never repeat execution.
        self.state.clear_pending_action()
        if self.response_service is not None:
            reply = self.response_service.respond(
                self._request_text, intent=plan.intent, tool_result=result,
            )
        elif result.success and result.action == "track_order":
            reply = (
                f"Your order {result.data['order_id']} "
                f"has status: {result.data['status']}."
            )
        else:
            reply = result.message or "The action could not be completed."

        # Finish this attempt; its messages and result remain in history.
        self.state.clear_pending_action()

        return self._record_reply(reply)

    def _record_reply(self, reply: str) -> str:
        self.state.add_assistant_message(reply)
        return reply