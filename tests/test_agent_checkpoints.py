import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from agent1.persistent_agent import PersistentAgent
from agent1.persistence.checkpoint_store import SQLiteCheckpointStore
from agent1.conversation.context_builder import ContextBuilder
from agent1.analysis.action_plan import ActionPlan
from agent1.routing.action_types import ActionType, RoutingDecision


def plan(multi=True):
    if not multi:
        return ActionPlan(request_type='single_action', intent='track_order',
                          action='order_service', reason='fixture')
    return ActionPlan.model_validate({'request_type':'multi_action','reason':'fixture','steps':[
        {'intent':'track_order','action':'order_service','arguments':{}},
        {'intent':'availability','action':'product_service',
         'arguments':{'product':'backpack','color':'blue'}}]})

class Analyzer:
    def __init__(self): self.contexts=[]
    def analyze(self, conversation_context, **kwargs):
        self.contexts.append(conversation_context)
        if 'hours' in conversation_context:
            return ActionPlan(request_type='general_response',intent='store_opening_hours',reason='fixture')
        return plan('backpack' in conversation_context)
    def classify_pending_reply(self, *args):
        return 'abandon' if args[-1]=='Never mind' else 'new_request'

class Responses:
    def __init__(self): self.text=None
    def reset_trace(self): pass
    def respond_many(self,text,records): self.text=text; return 'Two results'
    def respond(self,text,**kwargs): self.text=text; return 'Reply'

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'history.sqlite3'
    def agent(self):
        ensemble=SimpleNamespace(classes_=np.array(['track_order']),predict_proba=lambda _:np.array([[1.0]]))
        routing=RoutingDecision('track_order',1,ActionType.ORDER_SERVICE,1,1,[],False,'fixture')
        return PersistentAgent(ensemble_model=ensemble, router=SimpleNamespace(route=lambda *_:routing),
            analyzer=Analyzer(),context_builder=ContextBuilder(),response_service=Responses(),
            checkpoint_store=SQLiteCheckpointStore(self.path))
    def test_multi_resume_and_saved_completion(self):
        a=self.agent();a.handle_message('Track my order and check blue backpack')
        a.save_conversation('multi')
        b=self.agent();b.load_conversation('multi')
        self.assertIs(b.state,b.action_runner.state);self.assertIs(b.state,b.multi_action_runner.state)
        self.assertEqual(b.multi_action_runner.last_results,[])
        b.handle_message('ORD-123')
        self.assertEqual([r['intent'] for r in b.multi_action_runner.last_results],['track_order','availability'])
        self.assertTrue(all(r['result']['success'] for r in b.multi_action_runner.last_results))
        self.assertIn('blue backpack',b.response_service.text)
        self.assertIn('ORD-123',b.response_service.text)
        self.assertFalse(b.state.has_pending_action())
        b.save_conversation('multi')
        c=self.agent();c.load_conversation('multi')
        self.assertFalse(c.state.has_pending_action())
        self.assertEqual(c.multi_action_runner.last_results,[])
        self.assertEqual(c.state.messages,b.state.messages)
        self.assertEqual(sum(m.role.value=='tool' for m in c.state.messages),2)
    def test_single_resume(self):
        a=self.agent();a.handle_message('Track my order');a.save_conversation('single')
        b=self.agent().load_conversation('single');b.handle_message('ORD-123')
        self.assertTrue(b.action_runner.last_tool_result.success)
        self.assertIn('Track my order',b.response_service.text)
    def test_abandon_fresh_flag(self):
        a=self.agent();a.handle_message('Track my order');a.handle_message('Never mind')
        a.save_conversation('abandoned')
        b=self.agent().load_conversation('abandoned')
        self.assertTrue(b._fresh_request)
        b.handle_message('What are the store hours?')
        self.assertEqual(b.analyzer.contexts,['What are the store hours?'])
    def test_new_request_after_restore(self):
        a=self.agent();a.handle_message('Track my order and check blue backpack');a.save_conversation('x')
        b=self.agent().load_conversation('x');b.handle_message('What are the store hours?')
        self.assertFalse(b.state.has_pending_action())
        self.assertEqual(b.last_pending_decision,'new_request')
        self.assertEqual(b.multi_action_runner.last_results,[])
    def test_failed_restore_keeps_live_state(self):
        a=self.agent();a.state.add_user_message('Keep me');previous=a.state
        with self.assertRaises(KeyError):a.load_conversation('missing')
        self.assertIs(a.state,previous)
        a.save_conversation('bad')
        with sqlite3.connect(self.path) as con:
            con.execute('UPDATE agent_checkpoints SET checkpoint_json=?',('{"checkpoint_version":99}',))
        with self.assertRaises(ValueError):a.load_conversation('bad')
        self.assertIs(a.state,previous)
    def test_step1_rows_preserved(self):
        a=self.agent();a.checkpoint_store.save('old',a.state)
        a.state.add_user_message('new');a.save_conversation('old')
        self.assertEqual(len(a.checkpoint_store.load('old')),0)
        self.assertEqual(len(a.checkpoint_store.load_checkpoint('old')[0]),1)
    def test_clarification_preserved(self):
        a=self.agent();routing=RoutingDecision('track_order',.4,ActionType.ORDER_SERVICE,.4,.1,[],True,'test')
        a.state.add_clarification('Which operation?',routing);a.save_conversation('c')
        b=self.agent().load_conversation('c')
        self.assertEqual(b.state.pending_clarification,'Which operation?')
        self.assertEqual(b.state.pending_routing_decision,routing)
    def test_bad_runtime_preserves_checkpoint(self):
        a=self.agent();a.save_conversation('ok')
        with self.assertRaises(ValueError):a.checkpoint_store.save_checkpoint('ok',a.state,{'fresh_request':'bad'})
        self.assertEqual(len(a.checkpoint_store.load_checkpoint('ok')[0]),0)

if __name__=='__main__':unittest.main()
