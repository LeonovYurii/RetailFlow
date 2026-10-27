"""Sequential independent read-only execution, collecting all arguments first."""
import re
from agent1.analysis.action_plan import ActionPlan, RequestType
from agent1.tools.tool_result import ToolResult

class MultiActionRunner:
    REQUIREMENTS = {
        ("order_service", "track_order"): ("order_id",),
        ("product_service", "availability"): ("product", "color"),
    }

    def __init__(self, state, executor, response_service=None):
        self.state, self.executor = state, executor
        self.response_service = response_service
        self.last_results = []
        self.request_text = ""

    def start(self, plan, text):
        if self.state.has_pending_action():
            raise ValueError("Resolve the pending plan first.")
        self.last_results = []
        if any(step.depends_on or step.condition for step in plan.steps):
            return self._reply("Dependent or conditional tasks are not supported yet. Please request independent lookups.")
        required, arguments = [], {}
        for i, step in enumerate(plan.steps):
            for name in self.REQUIREMENTS.get((step.action.value, step.intent), ()):
                key = f"{i}.{name}"
                required.append(key)
                value = step.arguments.get(name)
                if isinstance(value, str) and value.strip():
                    value = value.strip()
                    if name == "order_id":
                        if not re.fullmatch(r"ORD-[0-9]+", value, re.I):
                            continue
                        value = value.upper()
                    arguments[key] = value

        # Patch: Handle requests consisting entirely of explicit tracking commands.
        tracking_pattern = (
            r"track\s+(?:order\s+)?(ORD-[0-9]+)"
        )
        clauses = re.split(
            r"\s+and\s+",
            text.strip().rstrip("."),
            flags=re.I,
        )
        matches = [
            re.fullmatch(tracking_pattern, clause.strip(), flags=re.I)
            for clause in clauses
        ]

        explicit_tracking = (
            len(clauses) == len(plan.steps)
            and all(matches)
            and all(
                step.intent == "track_order"
                and step.action.value == "order_service"
                for step in plan.steps
            )
        )

        if explicit_tracking:
            for i, match in enumerate(matches):
                supplied_id = match.group(1).upper()
                key = f"{i}.order_id"
                existing_id = arguments.get(key)

                if existing_id is not None and existing_id != supplied_id:
                    raise ValueError(
                        f"Task {i + 1}: model order ID conflicts "
                        "with the explicit tracking command."
                    )

                arguments[key] = supplied_id

        # Recover an omitted ID only when its step association is unique.
        order_steps = [
            (i, step)
            for i, step in enumerate(plan.steps)
            if step.action.value == "order_service"
        ]
        order_ids = list(dict.fromkeys(
            re.findall(r"\bORD-[0-9]+\b", text.upper())
        ))

        if len(order_steps) == 1 and len(order_ids) == 1:
            i, step = order_steps[0]
            key = f"{i}.order_id"

            if (
                step.intent == "track_order"
                and not step.arguments.get("order_id")
                and key not in arguments
            ):
                arguments[key] = order_ids[0]
        # end of the patch
        
        self.request_text = text
        self.state.start_pending_action(plan, required, arguments)
        return self._advance()

    def resume(self, text):
        missing = self.state.get_missing_arguments()
        if not missing:
            raise ValueError("No missing multi-action argument.")
        key = missing[0]
        i, name = key.split(".")
        if name == "order_id":
            ids = list(dict.fromkeys(re.findall(r"\bORD-[0-9]+\b", text.upper())))
            value = ids[0] if len(ids) == 1 else None
        else:
            # Only the explicitly requested field is filled. Semantic dispatch
            # has already decided that this reply continues the pending plan.
            value = text.strip()
        if value:
            self.state.update_pending_arguments(**{key: value})
        self.request_text += "\nFollow-up: " + text
        return self._advance()

    def _advance(self):
        plan = self.state.pending_action_plan
        missing = self.state.get_missing_arguments()
        if missing:
            i, name = missing[0].split(".")
            label = plan.steps[int(i)].intent
            hint = "one order number, such as ORD-123" if name == "order_id" else f"the {name} only"
            return self._reply(f"For task {int(i)+1} ({label}), please provide {hint}.")
        # Snapshot and clear before any execution or generation, preventing replay.
        args = dict(self.state.pending_arguments)
        self.state.clear_pending_action()
        self.last_results = []
        for i, step in enumerate(plan.steps):
            kwargs = {k.split(".", 1)[1]: v for k, v in args.items() if k.startswith(f"{i}.")}
            if (step.action.value, step.intent) not in self.REQUIREMENTS:
                result = ToolResult(success=False, action=step.intent,
                                    message=f"Task {i+1}: {step.intent} is not supported in multiple-action mode.")
            else:
                single = ActionPlan(request_type=RequestType.SINGLE_ACTION,
                                    intent=step.intent, action=step.action, reason=plan.reason)
                try:
                    result = self.executor.execute(single, **kwargs)
                except Exception as exc:
                    result = ToolResult(success=False, action=step.intent,
                                        message=f"Task {i+1} failed ({type(exc).__name__}); no result is available.")
            record = {"step_index": i, "intent": step.intent,
                      "arguments": kwargs, "result": result.model_dump()}
            self.last_results.append(record)
            self.state.add_tool_result(result.model_dump_json(), metadata={"step_index": i})
        from agent1.generation.response_service import format_multi_results
        reply = (self.response_service.respond_many(self.request_text, self.last_results)
                 if self.response_service is not None else format_multi_results(self.last_results))
        return self._reply(reply)

    def _reply(self, text):
        self.state.add_assistant_message(text)
        return text
