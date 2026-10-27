"""Step 1: explicit save/load of ConversationState, without model calls.

One row per conversation, containing the latest full JSON snapshot. Saving the
same ID replaces that snapshot transactionally. This is not an event log or an
Agent checkpoint: runner-local request text and Agent flags are not stored yet.
"""
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from agent1.analysis.action_plan import ActionPlan, RequestType
from agent1.conversation.state import ConversationState
from agent1.conversation.message import ChatMessage, MessageRole, MessageType
from agent1.routing.action_types import ActionType, ActionCandidate, RoutingDecision


def _encode(state):
    return {
        'schema_version': 1,
        'messages': [asdict(message) for message in state.messages],
        'pending_routing_decision': (
            asdict(state.pending_routing_decision)
            if state.pending_routing_decision is not None else None
        ),
        'pending_clarification': state.pending_clarification,
        'pending_action_plan': (
            state.pending_action_plan.model_dump(mode='json')
            if state.pending_action_plan is not None else None
        ),
        'pending_arguments': state.pending_arguments,
        'required_arguments': state.required_arguments,
    }


def _decode(data):
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported conversation snapshot version.')
    state = ConversationState()
    for record in data['messages']:
        message = dict(record)
        message['role'] = MessageRole(message['role'])
        message['message_type'] = MessageType(message['message_type'])
        state.messages.append(ChatMessage(**message))
    routing = data['pending_routing_decision']
    if routing is not None:
        routing = dict(routing)
        routing['action'] = ActionType(routing['action'])
        routing['candidates'] = [
            ActionCandidate(action=ActionType(c['action']), score=c['score'])
            for c in routing['candidates']
        ]
        state.pending_routing_decision = RoutingDecision(**routing)
    state.pending_clarification = data['pending_clarification']
    plan = data['pending_action_plan']
    if plan is not None:
        plan = ActionPlan.model_validate(plan)
        if plan.request_type not in (RequestType.SINGLE_ACTION, RequestType.MULTI_ACTION):
            raise ValueError('A pending action must be a resolved action plan.')
        if state.has_pending_clarification():
            raise ValueError('Snapshot contains both pending action and clarification.')
    arguments = data['pending_arguments']
    required = data['required_arguments']
    if not isinstance(arguments, dict):
        raise ValueError('Pending arguments must be a dictionary.')
    if not isinstance(required, list) or not all(isinstance(k, str) for k in required):
        raise ValueError('Required arguments must be a list of strings.')
    if plan is None and (arguments or required):
        raise ValueError('Pending arguments require a pending action plan.')
    state.pending_action_plan = plan
    state.pending_arguments = arguments
    state.required_arguments = required
    return state


class SQLiteConversationStore:
    """File-backed SQLite store; each operation opens and closes its connection.

    IDs are caller-owned nonempty strings. load() returns a new state or None.
    State metadata/arguments must contain JSON-compatible values. Last save wins;
    concurrent writers and interrupted tool execution are outside step 1.
    """

    def __init__(self, database_path):
        if str(database_path) == ':memory:':
            raise ValueError('Use a file path for persistence, not :memory:.')
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_snapshots (
                        conversation_id TEXT PRIMARY KEY,
                        snapshot_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

    @staticmethod
    def _check_id(conversation_id):
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError('conversation_id must be a nonempty string.')

    def save(self, conversation_id, state):
        self._check_id(conversation_id)
        if not isinstance(state, ConversationState):
            raise TypeError('state must be a ConversationState.')
        # Serialize and validate before writing, so errors cannot replace a good row.
        payload = json.dumps(_encode(state), ensure_ascii=False, allow_nan=False)
        _decode(json.loads(payload))
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute('''
                    INSERT INTO conversation_snapshots (conversation_id, snapshot_json)
                    VALUES (?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        snapshot_json = excluded.snapshot_json,
                        updated_at = CURRENT_TIMESTAMP
                ''', (conversation_id, payload))

    def load(self, conversation_id):
        self._check_id(conversation_id)
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                'SELECT snapshot_json FROM conversation_snapshots WHERE conversation_id = ?',
                (conversation_id,),
            ).fetchone()
        return None if row is None else _decode(json.loads(row[0]))
