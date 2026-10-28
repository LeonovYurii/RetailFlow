"""Listing support for the existing SQLite Agent checkpoint store."""
import json
import sqlite3
from contextlib import closing
from agent1.persistence.checkpoint_store import SQLiteCheckpointStore


def check_limit(limit):
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError('limit must be an integer between 1 and 200.')


def summary(conversation_id, created_at, updated_at, state):
    first = next((m['content'] for m in state['messages'] if m['role'] == 'user'), '')
    return dict(conversation_id=conversation_id, created_at=created_at,
                updated_at=updated_at, preview=' '.join(first.split())[:100],
                message_count=len(state['messages']),
                pending=state['pending_action_plan'] is not None or bool(state['pending_clarification']))


class SQLiteSessionStore(SQLiteCheckpointStore):
    def list_conversations(self, limit=20):
        check_limit(limit)
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute('''
                SELECT conversation_id, created_at, updated_at, checkpoint_json
                FROM agent_checkpoints
                ORDER BY updated_at DESC, conversation_id ASC LIMIT ?
            ''', (limit,)).fetchall()
        return [summary(cid, created, updated, json.loads(payload)['state'])
                for cid, created, updated, payload in rows]
