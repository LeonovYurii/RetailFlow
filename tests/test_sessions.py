import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from agent1.analysis.action_plan import ActionPlan
from agent1.persistent_agent import PersistentAgent
from agent1.conversation.context_builder import ContextBuilder
from agent1.routing.action_types import ActionType, RoutingDecision
from agent1.persistence.sqlite_session_store import SQLiteSessionStore
from agent1.persistence.mongo_store import MongoCheckpointStore
from agent1.session import ChatSession, UnsavedTurnError, SessionBlockedError
from agent1.chat_console import run_chat


def factory(store):
    p=ActionPlan.model_validate({'request_type':'multi_action','reason':'fixture','steps':[
        {'intent':'track_order','action':'order_service','arguments':{}},
        {'intent':'availability','action':'product_service','arguments':{'product':'backpack','color':'blue'}}]})
    class Analyzer:
        def analyze(self,**kwargs):return p.model_copy(deep=True)
        def classify_pending_reply(self,*args):return 'abandon'
    routing=RoutingDecision('track_order',1,ActionType.ORDER_SERVICE,1,1,[],False,'fixture')
    return PersistentAgent(
        ensemble_model=SimpleNamespace(classes_=np.array(['track_order']),predict_proba=lambda _:np.array([[1.0]])),
        router=SimpleNamespace(route=lambda *_:routing),analyzer=Analyzer(),
        context_builder=ContextBuilder(),checkpoint_store=store)

class Cursor(list):
    def sort(self, keys):
        for key,direction in reversed(keys):super().sort(key=lambda d:d[key],reverse=direction<0)
        return self
    def limit(self,n):return self[:n]

class FakeCollection:
    """API-shaped test double, not a MongoDB server or driver integration test."""
    write_concern=SimpleNamespace(acknowledged=True)
    def __init__(self):self.rows={}
    def update_one(self,query,update,upsert):
        key=query['_id'];new=key not in self.rows
        doc=self.rows.setdefault(key,{'_id':key})
        if new:doc.update(copy.deepcopy(update['$setOnInsert']))
        doc.update(copy.deepcopy(update['$set']))
        return SimpleNamespace(acknowledged=True)
    def find_one(self,query):return copy.deepcopy(self.rows.get(query['_id']))
    def find(self,query):return Cursor(copy.deepcopy(list(self.rows.values())))

class Sessions(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=SQLiteSessionStore(Path(self.tmp.name)/'history.db')
        self.session=ChatSession(self.store,factory);self.cid=self.session.new()
    def test_auto_save_resume_complete(self):
        self.session.send('Track my order and check blue backpack')
        new=ChatSession(self.store,factory);new.open(self.cid)
        self.assertEqual(new.agent.state.get_missing_arguments(),['0.order_id'])
        new.send('ORD-123')
        third=ChatSession(self.store,factory);third.open(self.cid)
        self.assertFalse(third.agent.state.has_pending_action())
        self.assertEqual(sum(m.role.value=='tool' for m in third.agent.state.messages),2)
    def test_abandon_autosaved(self):
        self.session.send('Track my order and check blue backpack');self.session.send('Never mind')
        restored=ChatSession(self.store,factory);restored.open(self.cid)
        self.assertFalse(restored.agent.state.has_pending_action());self.assertTrue(restored.agent._fresh_request)
    def test_list_and_isolation(self):
        self.session.send('First request');second=self.session.new()
        self.assertEqual(len(self.session.list_conversations()),2)
        self.assertEqual(self.session.history(),[])
        self.session.open(self.cid);self.assertTrue(self.session.history())
        with self.assertRaises(KeyError):self.session.open('absent')
        self.assertEqual(self.session.conversation_id,self.cid)
    def test_save_failure_no_reexecution(self):
        self.session.send('Track my order and check blue backpack')
        save=self.store.save_checkpoint
        def broken(*args):raise OSError('disk issue')
        self.store.save_checkpoint=broken
        with self.assertRaises(UnsavedTurnError) as caught:self.session.send('ORD-123')
        self.assertIn('shipped',caught.exception.answer)
        count=len(self.session.agent.state.messages)
        with self.assertRaises(SessionBlockedError):self.session.send('ORD-123')
        with self.assertRaises(SessionBlockedError):self.session.new()
        self.store.save_checkpoint=save
        self.session.retry_save()
        self.assertEqual(len(self.session.agent.state.messages),count)
        self.assertFalse(self.store.load_checkpoint(self.cid)[0].has_pending_action())
    def test_processing_failure_blocks_then_reload(self):
        def fail(text):self.session.agent.state.add_user_message(text);raise ValueError('bad plan')
        self.session.agent.handle_message=fail
        with self.assertRaises(ValueError):self.session.send('not saved')
        with self.assertRaises(SessionBlockedError):self.session.send('again')
        self.session.reload_saved();self.assertEqual(self.session.history(),[])
    def test_console(self):
        lines=iter(['Track my order and check blue backpack','ORD-123','/history','/list','/quit'])
        output=[];run_chat(self.session,lambda _:next(lines),output.append)
        self.assertTrue(any('shipped' in s for s in output))
        self.assertFalse(self.store.load_checkpoint(self.cid)[0].has_pending_action())
    def test_invalid_message_does_not_block(self):
        with self.assertRaises(ValueError):self.session.send(' ')
        self.session.send('Valid request')

class MongoAdapter(unittest.TestCase):
    def setUp(self):self.collection=FakeCollection();self.store=MongoCheckpointStore(self.collection)
    def test_session_contract(self):
        s=ChatSession(self.store,factory);cid=s.new();s.send('Track my order and check blue backpack')
        self.assertIsInstance(self.collection.rows[cid]['state']['pending_arguments'],list)
        other=ChatSession(self.store,factory);other.open(cid);other.send('ORD-123')
        state,runtime=self.store.load_checkpoint(cid)
        self.assertFalse(state.has_pending_action());self.assertEqual(len(s.list_conversations()),1)
        self.assertEqual(sum(m.role.value=='tool' for m in state.messages),2)
    def test_unknown_and_version(self):
        with self.assertRaises(KeyError):self.store.load_checkpoint('absent')
        s=ChatSession(self.store,factory);cid=s.new()
        self.collection.rows[cid]['checkpoint_version']=99
        with self.assertRaises(ValueError):self.store.load_checkpoint(cid)
    def test_preserve_created_time(self):
        s=ChatSession(self.store,factory);cid=s.new();created=self.collection.rows[cid]['created_at']
        s.send('Track my order')
        self.assertEqual(self.collection.rows[cid]['created_at'],created)
    def test_unacknowledged_rejected(self):
        self.collection.write_concern=SimpleNamespace(acknowledged=False)
        with self.assertRaises(ValueError):MongoCheckpointStore(self.collection)

if __name__=='__main__':unittest.main()
