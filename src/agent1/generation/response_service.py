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

    def respond_many(self, user_message, records):
        """Generate once from all step results; retain a complete factual fallback."""
        import json
        self.reset_trace()
        self.last_messages = build_response_messages(user_message, tool_result={"steps": records})
        self.last_messages[0]["content"] += (
            "\nAnswer every numbered task separately, including failures. "
            "Preserve step/order identifiers. Inventory marked mock_inventory is "
            "demo data, not live stock; explicitly label it mock inventory. "
            "Unknown inventory is not out of stock. Do not conflate independent results.")
        try:
            answer = self.generator.generate(self.last_messages, max_new_tokens=256)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Empty generated answer")
            self.last_mode = "v2_multi"
            # A visible label remains even if the model omits the mock qualifier.
            prefix = "Demo stock results use mock inventory.\n" if any(
                r["result"]["data"].get("source") == "mock_inventory" for r in records) else ""
            return prefix + answer.strip()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_mode = "generation_failed"
            return format_multi_results(records)


def format_multi_results(records):
    lines = []
    for record in records:
        result = record["result"]
        data = result["data"]
        if result["success"] and result["action"] == "track_order":
            text = f"Order {data['order_id']}: {data['status']}."
            if data.get("estimated_delivery"):
                text += f" Estimated delivery: {data['estimated_delivery']}."
        else:
            text = result.get("message") or "No result is available."
        lines.append(f"{record['step_index']+1}. {text}")
    return "\n".join(lines)
