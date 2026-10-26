import numpy as np

from agent1.analysis.action_plan import RequestType
from agent1.conversation import ConversationState
from agent1.execution.action_runner import ActionRunner
from agent1.routing.action_types import ActionType


class Agent:
    """Connect conversation, request analysis, and action execution."""

    def __init__(
        self,
        ensemble_model,
        router,
        analyzer,
        context_builder,
        top_n_intents: int = 5,
        response_service=None,
    ):
        self.ensemble_model = ensemble_model
        self.router = router
        self.analyzer = analyzer
        self.context_builder = context_builder
        self.top_n_intents = top_n_intents

        self.state = ConversationState()
        self.response_service = response_service
        self.action_runner = ActionRunner(self.state, response_service=response_service)

        # Expose the latest analysis for notebook inspection.
        self.last_routing_decision = None
        self.last_action_plan = None

    def handle_message(self, text: str) -> str:
        text = text.strip()

        if not text:
            raise ValueError("The message must not be empty.")

        if self.response_service is not None:
            self.response_service.reset_trace()
        self.action_runner.last_tool_result = None
        self.state.add_user_message(text)

        # Initial implementation: assume this reply supplies arguments
        # for the pending action. Intent changes will be handled next.
        if self.state.has_pending_action():
            return self.action_runner.resume(text)

        plan, routing_decision = self._analyze()

        if plan.request_type == RequestType.CLARIFICATION:
            question = plan.clarification_question
            if not question:
                raise ValueError("The clarification question is missing.")

            self.state.add_clarification(
                content=question,
                routing_decision=routing_decision,
            )
            return question

        self.state.resolve_clarification()

        if plan.request_type == RequestType.SINGLE_ACTION:
            if (
                plan.action == ActionType.ORDER_SERVICE
                and plan.intent == "track_order"
            ):
                return self.action_runner.start(plan, text)

            if self.response_service is not None:
                # Only informational intents enter knowledge generation.
                information_intents = {
                    "store_opening_hours", "store_location", "return_policy",
                    "refund_policy", "shipping_costs", "payment_methods",
                    "sales_period", "product_information", "customer_service",
                    "availability", "availability_in_store", "availability_online",
                    "recover_password",
                }
                if plan.intent in information_intents:
                    return self._reply(self.response_service.respond(text, intent=plan.intent))

            return self._reply(
                "This prototype currently supports order tracking. "
                "That action is not implemented yet."
            )

        if plan.request_type == RequestType.MULTI_ACTION:
            return self._reply(
                "I cannot process multiple actions together yet. "
                "Please send one request at a time."
            )

        if self.response_service is not None:
            return self._reply(self.response_service.respond(text, intent=plan.intent))

        # Backward-compatible response without a configured response service.
        return self._reply(
            "I can help you track an order. "
            "What would you like to check?"
        )

    def _analyze(self):
        context = self.context_builder.build_for_analyzer(self.state)

        probabilities = self.ensemble_model.predict_proba([context])[0]

        top_indices = np.argsort(probabilities)[::-1][
            :self.top_n_intents
        ]

        candidates_text = "\n".join(
            f"- {self.ensemble_model.classes_[idx]}: "
            f"{probabilities[idx]:.4f}"
            for idx in top_indices
        )

        routing_decision = self.router.route(
            probabilities,
            self.ensemble_model.classes_,
        )

        plan = self.analyzer.analyze(
            conversation_context=context,
            routing_decision=routing_decision,
            intent_candidates_text=candidates_text,
        )

        self.last_routing_decision = routing_decision
        self.last_action_plan = plan

        return plan, routing_decision

    def _reply(self, text: str) -> str:
        self.state.add_assistant_message(text)
        return text

    def reset(self):
        self.state.clear()
        self.action_runner.last_tool_result = None
        if self.response_service is not None:
            self.response_service.reset_trace()
        self.last_routing_decision = None
        self.last_action_plan = None