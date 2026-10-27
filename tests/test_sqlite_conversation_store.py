import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict
from agent1.persistence import SQLiteConversationStore
from agent1.conversation.state import ConversationState
from agent1.analysis.action_plan import ActionPlan
from agent1.routing.action_types import ActionType, ActionCandidate, RoutingDecision


def pending_state():
    state = ConversationState()
    state.add_user_message('Track my order and check the blue backpack.')
    state.add_assistant_message('Please provide your order number.')
    plan = ActionPlan.model_validate(dict(request_type='multi_action', reason='Two lookups', steps=[
        dict(intent='track_order', action='order_service', arguments={}),
        dict(intent='availability', action='product_service', arguments={'product':'backpack','color':'blue'}),
    ]))
    state.start_pending_action(plan, ['0.order_id','1.product','1.color'],
                               {'1.product':'backpack','1.color':'blue'})
    return state


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'history.db'
        self.store = SQLiteConversationStore(self.path)

    def test_pending_round_trip(self):
        original = pending_state()
        self.store.save('a', original)
        loaded = SQLiteConversationStore(self.path).load('a')
        self.assertEqual(loaded.pending_action_plan, original.pending_action_plan)
        self.assertEqual(loaded.messages, original.messages)
        self.assertEqual(loaded.pending_arguments, original.pending_arguments)
        self.assertEqual(loaded.get_missing_arguments(), ['0.order_id'])
        loaded.update_pending_arguments(**{'0.order_id':'ORD-123'})
        self.assertEqual(original.get_missing_arguments(), ['0.order_id'])
        self.assertEqual(self.store.load('a').get_missing_arguments(), ['0.order_id'])

    def test_tool_and_unicode(self):
        state = ConversationState()
        state.add_user_message('Привіт — café', {'nested':{'list':[1,True,None]}})
        state.add_tool_result('{"success":true}', {'step_index':0})
        self.store.save('unicode', state)
        self.assertEqual(self.store.load('unicode').messages, state.messages)

    def test_clarification_routing(self):
        state = ConversationState()
        routing = RoutingDecision('track_order', .4, ActionType.ORDER_SERVICE,.5,.3,
                    [ActionCandidate(ActionType.ORDER_SERVICE,.5)], False,'test')
        state.add_clarification('Track or view history?', routing)
        self.store.save('clarify', state)
        loaded = self.store.load('clarify')
        self.assertTrue(loaded.has_pending_clarification())
        self.assertEqual(loaded.pending_routing_decision,routing)
        self.assertEqual(loaded.messages,state.messages)

    def test_upsert_and_isolation(self):
        self.store.save('a',pending_state())
        self.store.save('b',ConversationState())
        state = self.store.load('a')
        state.clear_pending_action()
        self.store.save('a',state)
        self.assertFalse(self.store.load('a').has_pending_action())
        self.assertEqual(len(self.store.load('b')),0)
        self.assertIsNone(self.store.load('absent'))
        with sqlite3.connect(self.path) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM conversation_snapshots').fetchone()[0],2)

    def test_bad_save_preserves_row(self):
        self.store.save('a',pending_state())
        bad=ConversationState()
        bad.add_user_message('bad', {'unsupported':object()})
        with self.assertRaises(TypeError): self.store.save('a',bad)
        self.assertTrue(self.store.load('a').has_pending_action())

    def test_unknown_version(self):
        self.store.save('a',ConversationState())
        with sqlite3.connect(self.path) as con:
            con.execute('UPDATE conversation_snapshots SET snapshot_json=?', ('{"schema_version":99}',))
        with self.assertRaises(ValueError): self.store.load('a')

    def test_invalid_ids(self):
        for value in ('', ' ', None):
            with self.assertRaises(ValueError): self.store.save(value,ConversationState())
            with self.assertRaises(ValueError): self.store.load(value)

if __name__ == '__main__': unittest.main()
