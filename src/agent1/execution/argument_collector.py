import re

from agent1.analysis.action_plan import ActionPlan, RequestType
from agent1.routing.action_types import ActionType


class ArgumentCollector:
    """Collect execution arguments for supported actions."""

    ORDER_ID_PATTERN = re.compile(
        r"\bORD-[0-9]+\b",
        flags=re.IGNORECASE,
    )

    def required_arguments(
        self,
        action_plan: ActionPlan,
    ) -> list[str]:
        """Return argument names required by a supported action."""

        if (
            action_plan.request_type == RequestType.SINGLE_ACTION
            and action_plan.action == ActionType.ORDER_SERVICE
            and action_plan.intent == "track_order"
        ):
            return ["order_id"]

        raise ValueError(
            "Argument collection is not implemented for "
            f"{action_plan.action}/{action_plan.intent}."
        )

    def extract(
        self,
        text: str,
        action_plan: ActionPlan,
    ) -> dict[str, str]:
        """Extract an unambiguous order ID from the supplied text."""

        # Check that this action is supported.
        self.required_arguments(action_plan)

        order_ids = {
            match.upper()
            for match in self.ORDER_ID_PATTERN.findall(text)
        }

        # Avoid choosing arbitrarily when multiple IDs are mentioned.
        if len(order_ids) != 1:
            return {}

        return {"order_id": order_ids.pop()}