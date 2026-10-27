import re

import numpy as np

from agent1.analysis.action_plan import RequestType
from agent1.conversation import ConversationState
from agent1.execution.action_runner import ActionRunner
from agent1.execution.multi_action_runner import MultiActionRunner
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

        self.multi_action_runner = MultiActionRunner(
            self.state, self.action_runner.executor, response_service)

        # Expose the latest analysis for notebook inspection.
        self.last_routing_decision = None
        self.last_action_plan = None
        self.last_pending_decision = None
        self.last_pending_error = None
        self._fresh_request = False

    def handle_message(self, text: str) -> str:
        text = text.strip()

        if not text:
            raise ValueError("The message must not be empty.")

        if self.response_service is not None:
            self.response_service.reset_trace()
        self.action_runner.last_tool_result = None
        self.multi_action_runner.last_results = []
        self.state.add_user_message(text)

        self.last_pending_decision = None
        self.last_pending_error = None
        if self.state.has_pending_action():
            # Only a bare ID bypasses semantic analysis. An ID inside "cancel
            # ORD-123" must not accidentally execute the pending tracking task.
            missing = self.state.get_missing_arguments()
            waiting_for_order = bool(missing) and missing[0].split(".")[-1] == "order_id"
            if waiting_for_order and re.fullmatch(r"ORD-[0-9]+", text, flags=re.IGNORECASE):
                decision = "continue"
            else:
                try:
                    decision = self.analyzer.classify_pending_reply(
                        self.state.pending_action_plan,
                        self.state.get_missing_arguments(),
                        text,
                    )
                except ValueError as exc:
                    decision = "clarify"
                    self.last_pending_error = str(exc)
            self.last_pending_decision = decision
            if decision == "continue":
                runner = (self.multi_action_runner
                          if self.state.pending_action_plan.request_type == RequestType.MULTI_ACTION
                          else self.action_runner)
                return runner.resume(text)
            if decision == "abandon":
                self.state.clear_pending_action()
                self.state.resolve_clarification()
                self.last_action_plan = None
                self.last_routing_decision = None
                self._fresh_request = True
                return self._reply("Okay, I have stopped that request. No order was cancelled.")
            if decision == "new_request":
                # Analyze first so malformed analysis cannot discard the task.
                # Keep history stored, but exclude the abandoned task from this
                # new request's classifier/analyzer input.
                plan, routing_decision = self._analyze(current_request=text)
                self.state.clear_pending_action()
                self.state.resolve_clarification()
            else:
                if self.state.pending_action_plan.request_type == RequestType.MULTI_ACTION:
                    return self._reply("Would you like to continue these tasks, stop them, or ask a different question?")
                return self._reply(
                    "Would you like to continue tracking your order, stop that request, "
                    "or ask a different question?"
                )
        else:
            current_request = text if self._fresh_request else None
            plan, routing_decision = self._analyze(current_request=current_request)
            self._fresh_request = False

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
            return self.multi_action_runner.start(plan, text)

        if self.response_service is not None:
            return self._reply(self.response_service.respond(text, intent=plan.intent))

        # Backward-compatible response without a configured response service.
        return self._reply(
            "I can help you track an order. "
            "What would you like to check?"
        )

    def _analyze(self, current_request=None):
        context = (current_request if current_request is not None
                   else self.context_builder.build_for_analyzer(self.state))

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
        self.multi_action_runner.last_results = []
        if self.response_service is not None:
            self.response_service.reset_trace()
        self.last_routing_decision = None
        self.last_action_plan = None
        self.last_pending_decision = None
        self.last_pending_error = None
        self._fresh_request = False