from agent1.tools.tool_result import ToolResult


class OrderService:
    """Mock service for operations related to customer orders."""

    def __init__(self):
        self.orders = {
            "ORD-123": {
                "status": "shipped",
                "estimated_delivery": "2026-09-23",
            },
            "ORD-456": {
                "status": "processing",
                "estimated_delivery": "2026-09-26",
            },
        }

    def track_order(self, order_id: str) -> ToolResult:
        """Return tracking information for an order."""

        order = self.orders.get(order_id)

        if order is None:
            return ToolResult(
                success=False,
                action="track_order",
                message=f"Order {order_id} was not found.",
            )

        return ToolResult(
            success=True,
            action="track_order",
            data={
                "order_id": order_id,
                **order,
            },
        )