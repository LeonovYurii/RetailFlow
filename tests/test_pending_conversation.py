import unittest
from test_v2_integration import Generator, Knowledge, Classifier, plan
from agent1.agent import Agent
from agent1.analysis.request_analyzer import RequestAnalyzer
from agent1.conversation.context_builder import ContextBuilder
from agent1.generation.response_service import ResponseService
from agent1.routing.intent_router import IntentRouter
from agent1.routing.intent_mapping import INTENT_TO_ACTION
from agent1.routing.action_types import ActionType

class Analyzer:
    def __init__(self, decision='continue', next_plan=None):
        self.decision=decision; self.next_plan=next_plan
        self.inputs=[]; self.pending_calls=0
    def analyze(self, **kwargs):
        self.inputs.append(kwargs['conversation_context'])
        if len(self.inputs)==1: return plan()
        if isinstance(self.next_plan, Exception): raise self.next_plan
        return self.next_plan or plan()
    def classify_pending_reply(self, *args):
        self.pending_calls += 1
        if isinstance(self.decision, Exception): raise self.decision
        return self.decision

class PendingTests(unittest.TestCase):
    def make(self, decision='continue', next_plan=None):
        analyzer=Analyzer(decision,next_plan)
        service=ResponseService(Generator(), Knowledge())
        a=Agent(Classifier(),IntentRouter(INTENT_TO_ACTION),analyzer,ContextBuilder(),response_service=service)
        a.handle_message('Where is my order?')
        self.assertTrue(a.state.has_pending_action())
        return a
    def test_bare_id_fast_path(self):
        a=self.make();a.handle_message('ord-123')
        self.assertEqual(a.analyzer.pending_calls,0)
        self.assertEqual(a.last_pending_decision,'continue')
        self.assertFalse(a.state.has_pending_action())
        self.assertTrue(a.action_runner.last_tool_result.success)
        self.assertEqual(len(a.state),5)
    def test_abandon(self):
        a=self.make('abandon');reply=a.handle_message('Never mind')
        self.assertIn('No order was cancelled',reply)
        self.assertFalse(a.state.has_pending_action())
        self.assertIsNone(a.last_action_plan)
        self.assertIsNone(a.action_runner.last_tool_result)
        self.assertFalse(a.response_service.generator.calls)
        self.assertEqual(len(a.state),4)
    def test_topic_switch_excludes_stale_context(self):
        a=self.make('new_request',plan('store_opening_hours',ActionType.STORE_INFO))
        text='Actually, what time does Central close?';a.handle_message(text)
        self.assertEqual(a.analyzer.inputs[-1],text)
        self.assertFalse(a.state.has_pending_action())
        self.assertIsNone(a.action_runner.last_tool_result)
        self.assertIn('Central closes',a.response_service.last_messages[-1]['content'])
        self.assertEqual(len(a.state),4)
    def test_ambiguous_reply_keeps_task(self):
        a=self.make('clarify');reply=a.handle_message('Yes')
        self.assertIn('continue tracking',reply)
        self.assertTrue(a.state.has_pending_action())
        self.assertIsNone(a.action_runner.last_tool_result)
        a.handle_message('ORD-456')
        self.assertFalse(a.state.has_pending_action())
        self.assertEqual(a.action_runner.last_tool_result.data['order_id'],'ORD-456')
    def test_id_in_different_request_does_not_track(self):
        a=self.make('new_request',plan('cancel_order'))
        reply=a.handle_message('Cancel ORD-123')
        self.assertEqual(a.analyzer.pending_calls,1)
        self.assertIn('not implemented',reply)
        self.assertIsNone(a.action_runner.last_tool_result)
        self.assertFalse(a.response_service.generator.calls)
    def test_invalid_pending_decision_keeps_task(self):
        a=self.make(ValueError('bad JSON'));a.handle_message('something')
        self.assertEqual(a.last_pending_decision,'clarify')
        self.assertEqual(a.last_pending_error,'bad JSON')
        self.assertTrue(a.state.has_pending_action())
    def test_new_request_analysis_failure_preserves_task(self):
        a=self.make('new_request',ValueError('invalid action plan'))
        with self.assertRaises(ValueError):a.handle_message('When does Central close?')
        self.assertTrue(a.state.has_pending_action())
        self.assertIsNone(a.action_runner.last_tool_result)
    def test_after_abandon_new_request_starts_clean(self):
        a=self.make('abandon',plan('store_opening_hours',ActionType.STORE_INFO))
        a.handle_message('Never mind');a.handle_message('Central hours?')
        self.assertEqual(a.analyzer.inputs[-1],'Central hours?')
    def test_multiple_ids_stay_pending(self):
        a=self.make();a.handle_message('ORD-123 or ORD-456')
        self.assertTrue(a.state.has_pending_action())
        self.assertIsNone(a.action_runner.last_tool_result)
    def test_reset_clears_decision(self):
        a=self.make('clarify');a.handle_message('Yes');a.reset()
        self.assertIsNone(a.last_pending_decision)
        self.assertIsNone(a.last_pending_error)
        self.assertEqual(len(a.state),0)
    def test_real_pending_parser(self):
        for raw in ['{"decision":"abandon"}', '```json\n{"decision":"abandon"}\n```']:
            a=RequestAnalyzer(lambda s,u:raw)
            self.assertEqual(a.classify_pending_reply(plan(),['order_id'],'Never mind'),'abandon')
        for raw in ['bad', '{}', '[]', '{"decision":"cancel_order"}']:
            a=RequestAnalyzer(lambda s,u:raw)
            with self.assertRaises(ValueError):a.classify_pending_reply(plan(),['order_id'],'Yes')

if __name__=='__main__':unittest.main()
