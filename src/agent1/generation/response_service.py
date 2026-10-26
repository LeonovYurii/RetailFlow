"""Grounded response generation shared by knowledge and action flows."""
from agent1.generation.response_prompt import build_response_messages


class ResponseService:
    STOCK_INTENTS = {"availability", "availability_in_store", "availability_online"}

    def __init__(self, generator, knowledge_store):
        self.generator = generator
        self.knowledge_store = knowledge_store
        self.last_messages = None
        self.last_mode = None
        self.last_error = None

    def reset_trace(self):
        self.last_messages = None
        self.last_mode = None
        self.last_error = None

    def respond(self, user_message, intent=None, tool_result=None):
        self.reset_trace()
        # No stock tool is connected. Unknown inventory must not mean out of stock.
        if intent in self.STOCK_INTENTS:
            self.last_mode = "stock_information_unavailable"
            return "I do not have live stock information, so I cannot confirm whether that item is in stock."
        context = self.knowledge_store.get_context(intent)
        if tool_result is None and not context:
            self.last_mode = "no_source"
            return "I do not have the information needed to answer that request yet."
        self.last_messages = build_response_messages(
            user_message=user_message, knowledge_context=context,
            tool_result=tool_result.model_dump() if tool_result is not None else None,
        )
        try:
            answer = self.generator.generate(self.last_messages, max_new_tokens=128)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Empty generated answer")
            self.last_mode = "v2"
            return answer.strip()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            # Preserve completed execution even when response inference fails.
            self.last_mode = "generation_failed"
            if tool_result is not None:
                if tool_result.success and tool_result.action == "track_order":
                    data = tool_result.data
                    status = str(data.get("status", "unknown")).replace("_", " ")
                    reply = f"Order {data.get('order_id', '')} status: {status}."
                    eta = data.get("estimated_delivery")
                    return reply + (f" Estimated delivery: {eta}." if eta else " Estimated delivery is unavailable.")
                return tool_result.message or "The action could not be completed."
            return "I could not generate an answer just now. Please try again."
