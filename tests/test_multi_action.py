"""Offline contract/control-flow tests. No model weights or GPU required."""
import json
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from test_v2_integration import Classifier, Generator, Knowledge, plan
from agent1.agent import Agent
from agent1.analysis.action_plan import ActionPlan, ActionStep
from agent1.analysis.request_analyzer import RequestAnalyzer
from agent1.analysis.request_type_normalizer import RequestTypeNormalizer
from agent1.conversation.context_builder import ContextBuilder
from agent1.generation.response_service import ResponseService
from agent1.routing.intent_router import IntentRouter
from agent1.routing.intent_mapping import INTENT_TO_ACTION
from agent1.tools.stock_service import StockService


def track(order_id='ORD-123'):
    return ActionStep(intent='track_order', action='order_service', arguments={'order_id': order_id})

def stock(product='backpack', color='blue'):
    return ActionStep(intent='availability', action='product_service', arguments={'product': product, 'color': color})

def multi(*steps):
    return ActionPlan(request_type='multi_action', steps=list(steps or (track(), stock())), reason='test')

class Analyzer:
    def __init__(self, p): self.p=p; self.decision='continue'; self.next_plan=None
    def analyze(self, **kwargs):
        if isinstance(self.next_plan, Exception): raise self.next_plan
        return self.next_plan or self.p
    def classify_pending_reply(self, *args): return self.decision

class MultiActionTests(unittest.TestCase):
    def make(self, p=None, generate=False):
        service=ResponseService(Generator(), Knowledge()) if generate else None
        return Agent(Classifier(), IntentRouter(INTENT_TO_ACTION), Analyzer(p or multi()),
                     ContextBuilder(), response_service=service)
    def test_two_lookups(self):
        a=self.make();reply=a.handle_message('Track ORD-123 and check blue backpack')
        self.assertIn('shipped',reply);self.assertIn('Mock inventory',reply)
        self.assertEqual([r['result']['success'] for r in a.multi_action_runner.last_results],[True,True])
        self.assertFalse(a.state.has_pending_action());self.assertEqual(len(a.state),4)
    def test_two_orders_same_service(self):
        a=self.make(multi(track(),track('ORD-456')));reply=a.handle_message('Track both orders')
        self.assertIn('ORD-123',reply);self.assertIn('ORD-456',reply)
        self.assertEqual([r['arguments']['order_id'] for r in a.multi_action_runner.last_results],['ORD-123','ORD-456'])
    def test_partial_failure(self):
        a=self.make(multi(track('ORD-999'),stock()));reply=a.handle_message('Both')
        self.assertIn('not found',reply);self.assertIn('5 available',reply)
        self.assertEqual([r['result']['success'] for r in a.multi_action_runner.last_results],[False,True])
    def test_missing_argument_no_early_execution(self):
        a=self.make(multi(track(None),stock()));a.handle_message('Track and stock')
        self.assertTrue(a.state.has_pending_action());self.assertEqual(a.multi_action_runner.last_results,[])
        a.handle_message('ORD-123')
        self.assertEqual(len(a.multi_action_runner.last_results),2)
        self.assertEqual(a.last_pending_decision,'continue')
    def test_two_missing_ids_fill_separately(self):
        a=self.make(multi(track(None),track(None)));a.handle_message('Track two orders')
        reply=a.handle_message('ORD-123');self.assertIn('task 2',reply)
        self.assertEqual(a.multi_action_runner.last_results,[])
        a.handle_message('ORD-456')
        self.assertEqual([r['arguments']['order_id'] for r in a.multi_action_runner.last_results],['ORD-123','ORD-456'])
    def test_ambiguous_ids_not_assigned(self):
        a=self.make(multi(track(None),stock()));a.handle_message('Both');a.handle_message('ORD-123 or ORD-456')
        self.assertTrue(a.state.has_pending_action());self.assertEqual(a.multi_action_runner.last_results,[])
    def test_invalid_initial_id_prompts(self):
        a=self.make(multi(track('123'),stock()));reply=a.handle_message('Both')
        self.assertIn('order number',reply);self.assertTrue(a.state.has_pending_action())
    def test_missing_color(self):
        a=self.make(multi(track(),stock(color=None)));reply=a.handle_message('Both')
        self.assertIn('color only',reply);a.handle_message('red')
        self.assertEqual(a.multi_action_runner.last_results[1]['result']['data']['quantity'],0)
    def test_unknown_stock_not_zero_stock(self):
        a=self.make(multi(track(),stock('jacket')));reply=a.handle_message('Both')
        r=a.multi_action_runner.last_results[1]['result']
        self.assertFalse(r['success']);self.assertNotIn('quantity',r['data']);self.assertIn('unknown',reply)
    def test_abandon(self):
        a=self.make(multi(track(None),stock()));a.handle_message('Both');a.analyzer.decision='abandon'
        a.handle_message('Never mind');self.assertFalse(a.state.has_pending_action())
        self.assertEqual(a.multi_action_runner.last_results,[])
    def test_switch(self):
        a=self.make(multi(track(None),stock()));a.handle_message('Both')
        a.analyzer.decision='new_request';a.analyzer.next_plan=plan('cancel_order')
        reply=a.handle_message('Cancel ORD-123');self.assertIn('not implemented',reply)
        self.assertFalse(a.state.has_pending_action());self.assertEqual(a.multi_action_runner.last_results,[])
    def test_failed_switch_keeps_plan(self):
        a=self.make(multi(track(None),stock()));a.handle_message('Both')
        a.analyzer.decision='new_request';a.analyzer.next_plan=ValueError('bad plan')
        with self.assertRaises(ValueError):a.handle_message('Other task')
        self.assertEqual(len(a.state.pending_action_plan.steps),2)
    def test_ambiguous_reply_and_reset(self):
        a=self.make(multi(track(None),stock()));a.handle_message('Both');a.analyzer.decision='clarify'
        self.assertIn('continue these tasks',a.handle_message('Yes'));self.assertTrue(a.state.has_pending_action())
        a.reset();self.assertFalse(a.state.has_pending_action());self.assertEqual(a.multi_action_runner.last_results,[])
    def test_unsupported_step_not_executed(self):
        cancel=ActionStep(intent='cancel_order',action='order_service',arguments={'order_id':'ORD-123'})
        a=self.make(multi(cancel,stock()));reply=a.handle_message('Cancel and stock')
        self.assertIn('not supported',reply);self.assertIn('5 available',reply)
        self.assertIn('ORD-123',a.action_runner.executor.order_service.orders)
    def test_dependencies_not_executed(self):
        s=stock();s.depends_on=[0];a=self.make(multi(track(),s))
        self.assertIn('not supported',a.handle_message('Dependent tasks'))
        self.assertEqual(a.multi_action_runner.last_results,[])
    def test_tool_exception_does_not_stop_next_step(self):
        a=self.make()
        with patch.object(a.action_runner.executor.order_service,'track_order',side_effect=RuntimeError('offline')):
            reply=a.handle_message('Both')
        self.assertIn('failed',reply);self.assertTrue(a.multi_action_runner.last_results[1]['result']['success'])
    def test_generation_called_once_with_both_results(self):
        a=self.make(generate=True);a.handle_message('Both')
        self.assertEqual(len(a.response_service.generator.calls),1)
        content=a.response_service.last_messages[-1]['content']
        self.assertIn('ORD-123',content);self.assertIn('mock_inventory',content)
        self.assertEqual(a.response_service.last_mode,'v2_multi')
    def test_generation_failure_fallback_preserves_all_results(self):
        a=self.make(generate=True);a.response_service.generator.fail=True
        with patch.object(a.action_runner.executor,'execute',wraps=a.action_runner.executor.execute) as execute:
            reply=a.handle_message('Both');self.assertEqual(execute.call_count,2)
        self.assertIn('ORD-123',reply);self.assertIn('Mock inventory',reply)
        self.assertFalse(a.state.has_pending_action());self.assertEqual(len(a.multi_action_runner.last_results),2)
    def test_schema_and_real_analyzer_parser(self):
        p=multi(track(),track('ORD-456'))
        parsed=RequestAnalyzer(lambda s,u:p.model_dump_json())._parse_response(p.model_dump_json())
        self.assertEqual(len(parsed.steps),2)
        with self.assertRaises(ValidationError):
            ActionPlan(request_type='multi_action', candidate_actions=['order_service','product_service'],reason='old format')
        with self.assertRaises(ValidationError):multi(track())
        with self.assertRaises(ValidationError):
            ActionPlan(request_type='single_action',action='order_service',steps=[track()],reason='mixed')
    def test_normalizer_uses_steps(self):
        n=RequestTypeNormalizer(encode=lambda x:None)
        with patch.object(n,'_semantic_match',return_value=(multi().request_type,1.0)):
            data=multi().model_dump(mode='json');data['request_type']='multiple operations'
            self.assertEqual(n.normalize(data)['request_type'],'multi_action')
    def test_pending_classifier_receives_whole_plan(self):
        calls=[]
        def llm(s,u):calls.append(json.loads(u));return '{"decision":"continue"}'
        RequestAnalyzer(llm).classify_pending_reply(multi(track(None),stock()),['0.order_id'],'ORD-123')
        self.assertEqual(len(calls[0]['pending_steps']),2)

if __name__=='__main__':unittest.main()
