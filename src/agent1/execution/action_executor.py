from agent1.tools.order_service import OrderService
from agent1.tools.tool_result import ToolResult
from agent1.analysis.action_plan import ActionPlan, RequestType
from agent1.routing.action_types import ActionType


class ActionExecutor:
    """Execute resolved actions produced by the request analyzer."""

    def __init__(self):
        self.order_service = OrderService()

    def execute(
        self,
        action_plan: ActionPlan,
        **kwargs,
    ) -> ToolResult:
        """Execute a single resolved action."""

        if action_plan.request_type != RequestType.SINGLE_ACTION:
            return ToolResult(
                success=False,
                action=action_plan.intent or "none",
                message=(
                    f"Cannot execute request type "
                    f"{action_plan.request_type.value}."
                ),
            )

        if action_plan.action == ActionType.ORDER_SERVICE:
            return self._execute_order_action(
                action_plan.intent,
                **kwargs,
            )

        return ToolResult(
            success=False,
            action=action_plan.intent or "unknown",
            message=f"Unsupported action: {action_plan.action}",
        )

    def _execute_order_action(
        self,
        intent: str | None,
        **kwargs,
    ) -> ToolResult:

        if intent == "track_order":
            order_id = kwargs.get("order_id")

            if order_id is None:
                return ToolResult(
                    success=False,
                    action="track_order",
                    message="order_id is required.",
                )

            if not isinstance(order_id, str):
                return ToolResult(
                    success=False,
                    action="track_order",
                    message="order_id must be a string.",
                )

            order_id = order_id.strip()

            if not order_id:
                return ToolResult(
                    success=False,
                    action="track_order",
                    message="order_id is required.",
                )

            return self.order_service.track_order(order_id)

        return ToolResult(
            success=False,
            action=intent or "unknown",
            message=f"Unsupported order intent: {intent}",
        )