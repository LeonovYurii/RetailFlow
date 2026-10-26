import json
import unittest
import numpy as np
from agent1.agent import Agent
from agent1.analysis.action_plan import ActionPlan, RequestType
from agent1.conversation.context_builder import ContextBuilder
from agent1.conversation.state import ConversationState
from agent1.execution.action_runner import ActionRunner
from agent1.generation.response_service import ResponseService
from agent1.routing.action_types import ActionType
from agent1.routing.intent_router import IntentRouter
from agent1.routing.intent_mapping import INTENT_TO_ACTION

class Generator:
    def __init__(self): self.calls = []; self.fail = False
    def generate(self, messages, max_new_tokens):
        self.calls.append(messages)
        if self.fail: raise RuntimeError("test failure")
        return "Generated grounded answer."

class Knowledge:
    def get_context(self, intent):
        return "Central closes at 18:00." if intent == "store_opening_hours" else ""

class Classifier:
    classes_ = np.array(["track_order", "store_opening_hours"])
    def predict_proba(self, texts): return np.array([[0.9, 0.1]])

class Analyzer:
    def __init__(self, plan): self.plan = plan; self.calls = 0
    def analyze(self, **kwargs): self.calls += 1; return self.plan
    def classify_pending_reply(self, *args): return "continue"

def plan(intent="track_order", action=ActionType.ORDER_SERVICE):
    return ActionPlan(request_type=RequestType.SINGLE_ACTION, intent=intent, action=action, reason="test")

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.generator = Generator()
        self.service = ResponseService(self.generator, Knowledge())
    def agent(self, p):
        return Agent(Classifier(), IntentRouter(INTENT_TO_ACTION), Analyzer(p), ContextBuilder(), response_service=self.service)
    def test_missing_id_then_execution_once(self):
        a = self.agent(plan())
        self.assertIn("order number", a.handle_message("Where is my order?"))
        self.assertFalse(self.generator.calls)
        self.assertTrue(a.state.has_pending_action())
        self.assertEqual(a.handle_message("ORD-123"), "Generated grounded answer.")
        self.assertEqual(a.analyzer.calls, 1)
        self.assertFalse(a.state.has_pending_action())
        self.assertEqual(len(a.state.get_messages()), 5)
        prompt = self.generator.calls[-1][-1]["content"]
        self.assertIn("Where is my order?", prompt)
        self.assertIn('"status": "shipped"', prompt)
        self.assertIn("ORD-123", prompt)
    def test_ambiguous_ids_do_not_execute(self):
        a=self.agent(plan());a.handle_message("ORD-123 or ORD-456")
        self.assertTrue(a.state.has_pending_action());self.assertIsNone(a.action_runner.last_tool_result)
        self.assertFalse(self.generator.calls)
    def test_unknown_order_is_failed_tool_result(self):
        a=self.agent(plan());a.handle_message("Track ORD-999")
        self.assertFalse(a.action_runner.last_tool_result.success)
        self.assertIn('"success": false', self.generator.calls[-1][-1]["content"])
    def test_known_information_flows_to_generator(self):
        a=self.agent(plan("store_opening_hours", ActionType.STORE_INFO))
        a.handle_message("When does Central close?")
        self.assertIn("Central closes at 18:00", self.generator.calls[-1][-1]["content"])
        self.assertEqual(len(a.state.get_messages()), 2)
    def test_general_information_retains_intent(self):
        p=ActionPlan(request_type=RequestType.GENERAL_RESPONSE, intent="store_opening_hours", reason="info")
        self.agent(p).handle_message("When does Central close?")
        self.assertEqual(self.service.last_mode,"v2")
    def test_stock_unknown_is_not_out_of_stock(self):
        for intent in self.service.STOCK_INTENTS:
            a=self.agent(plan(intent, ActionType.PRODUCT_SEARCH))
            answer=a.handle_message("Is the blue jacket available?")
            self.assertIn("cannot confirm", answer)
            self.assertNotIn("is not available", answer)
        self.assertFalse(self.generator.calls)
    def test_unsupported_cancellation_does_not_execute(self):
        a=self.agent(plan("cancel_order"));answer=a.handle_message("Cancel ORD-123")
        self.assertIn("not implemented", answer)
        self.assertIsNone(a.action_runner.last_tool_result);self.assertFalse(self.generator.calls)
    def test_generation_failure_preserves_tool_and_clears_pending(self):
        self.generator.fail=True;a=self.agent(plan());answer=a.handle_message("Track ORD-123")
        self.assertIn("shipped",answer);self.assertFalse(a.state.has_pending_action())
        self.assertEqual(len(a.state.get_messages()),3)
        self.assertEqual(self.service.last_mode,"generation_failed")
        self.assertIn("test failure",self.service.last_error)
    def test_clarification_and_multi_action_do_not_generate(self):
        p=ActionPlan(request_type=RequestType.CLARIFICATION, needs_clarification=True, clarification_question="Which operation?",reason="test")
        a=self.agent(p);self.assertEqual(a.handle_message("Help"),"Which operation?")
        self.assertTrue(a.state.has_pending_clarification())
        p=ActionPlan(request_type=RequestType.MULTI_ACTION,candidate_actions=[ActionType.ORDER_SERVICE,ActionType.PRODUCT_SEARCH],reason="test")
        self.assertIn("one request",self.agent(p).handle_message("Track and find stock"))
        self.assertFalse(self.generator.calls)
    def test_no_source_does_not_generate_facts(self):
        a=self.agent(ActionPlan(request_type=RequestType.GENERAL_RESPONSE,reason="unknown"))
        self.assertIn("do not have",a.handle_message("What is the weather?"));self.assertFalse(self.generator.calls)
    def test_reset_clears_state_and_trace(self):
        a=self.agent(plan());a.handle_message("Track ORD-123");a.reset()
        self.assertEqual(len(a.state),0);self.assertIsNone(self.service.last_messages)
        self.assertIsNone(a.action_runner.last_tool_result)
    def test_original_runner_still_works_without_generator(self):
        s=ConversationState();r=ActionRunner(s);s.add_user_message("Track ORD-456")
        self.assertIn("processing",r.start(plan(),"Track ORD-456"))
        self.assertEqual(len(s),3)

if __name__ == "__main__": unittest.main()
